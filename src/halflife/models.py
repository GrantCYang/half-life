"""Optional Transformer execution. Importing this module never loads a model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def render_candidate(
    tokenizer: Any, prompt: str, answer: str
) -> tuple[list[int], int, int]:
    """Return full chat ids, prompt length and answer-only endpoint.

    Prefix checks prevent silently scoring a different tokenization. The full
    suffix includes the assistant end marker and any template newline.
    """
    if answer not in ("ALIVE", "DEAD"):
        raise ValueError("answer must be ALIVE or DEAD")
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(prefix + answer, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if not prefix_ids or answer_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "chat prompt is not a token prefix of answer; incompatible template"
        )
    if full_ids[: len(answer_ids)] != answer_ids or len(answer_ids) == len(prefix_ids):
        raise ValueError("answer is not a nonempty prefix of full assistant reply")
    return full_ids, len(prefix_ids), len(answer_ids)


def candidate_probability(
    alive_log_likelihood: float, dead_log_likelihood: float
) -> float:
    """Two-candidate normalized likelihood, not whole-vocabulary probability."""
    if not all(math.isfinite(x) for x in (alive_log_likelihood, dead_log_likelihood)):
        raise ValueError("candidate log likelihoods must be finite")
    difference = alive_log_likelihood - dead_log_likelihood
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    ratio = math.exp(difference)
    return ratio / (1.0 + ratio)


def _load_model(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attention: str,
    local_files_only: bool = True,
):
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(
        model_path, local_files_only=local_files_only
    )
    # Qwen3.8-27B uses this architecture. Exact archived weight identity is
    # pinned separately in configs/model_metadata.json and weight_identity.json.
    if config.model_type == "qwen3_5":
        model_class = transformers.Qwen3_5ForConditionalGeneration
    else:
        model_class = transformers.AutoModelForImageTextToText
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    model = model_class.from_pretrained(
        model_path,
        dtype=torch_dtype,
        attn_implementation=attention,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


class Forecaster:
    """Score ALIVE and DEAD independently with teacher-forced likelihoods."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        attention: str = "eager",
        max_length: int = 1024,
    ):
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = _load_model(
            model_path, device=device, dtype=dtype, attention=attention
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=False
            )
        self.model.eval()
        self.device = device
        self.max_length = max_length

    def score(self, prompt: str, mode: str = "full") -> dict[str, Any]:
        import torch

        if mode not in {"full", "label_only"}:
            raise ValueError("mode must be full or label_only")
        totals, details = {}, {}
        with torch.inference_mode():
            for answer in ("ALIVE", "DEAD"):
                ids, start, answer_end = render_candidate(
                    self.tokenizer, prompt, answer
                )
                end = len(ids) if mode == "full" else answer_end
                if len(ids) > self.max_length:
                    raise ValueError(
                        f"{len(ids)} tokens exceed max_length={self.max_length}; no truncation"
                    )
                inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits = (
                    self.model(
                        input_ids=inputs,
                        attention_mask=torch.ones_like(inputs),
                        use_cache=False,
                    )
                    .logits[0, start - 1 : end - 1]
                    .float()
                )
                targets = inputs[0, start:end]
                token_logs = (
                    torch.log_softmax(logits, dim=-1)
                    .gather(1, targets[:, None])
                    .squeeze(1)
                )
                totals[answer] = float(token_logs.sum().item())
                details[answer] = {
                    "token_ids": ids[start:end],
                    "token_log_probabilities": token_logs.cpu().tolist(),
                }
        return {
            "p_alive_raw": candidate_probability(totals["ALIVE"], totals["DEAD"]),
            "scoring_mode": mode,
            "candidate_log_likelihoods": totals,
            "candidate_tokens": details,
            "prediction_origin": "new_model_inference",
        }


class VisualInstrument:
    """Frozen multi-image chat generator used by Observer and Interrogator."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        attention: str = "eager",
    ):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = _load_model(
            model_path, device=device, dtype=dtype, attention=attention
        ).eval()
        self.device = device

    def generate(
        self, image_paths: list[str | Path], prompt: str, max_new_tokens: int
    ) -> str:
        import torch
        from PIL import Image

        images = []
        for path in image_paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        if not images:
            raise ValueError("at least one frame is required")
        content = [{"type": "image"} for _ in images] + [
            {"type": "text", "text": prompt}
        ]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens
            )
        suffix = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def parse_observer_output(text: str) -> dict[str, list[str]]:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(stripped)
    if not isinstance(result, dict) or set(result) != {"visible", "motion", "inferred"}:
        raise ValueError("Observer must return visible, motion, inferred lists")
    for key, values in result.items():
        if not isinstance(values, list) or any(
            not isinstance(x, str) or not x.strip() for x in values
        ):
            raise ValueError(f"invalid Observer {key} list")
    if not result["visible"] or len(result["inferred"]) > 3:
        raise ValueError("Observer needs visible facts and at most three hypotheses")
    return result


def parse_interrogator_output(text: str) -> str:
    answer = text.strip().upper()
    if answer not in {"STILL_TRUE", "NO_LONGER_TRUE", "CANNOT_TELL"}:
        raise ValueError(f"invalid Interrogator reply: {answer!r}")
    return answer


class CaptionEntailment:
    """NLI confirmation: future caption as premise, queried fact as hypothesis."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        entailment_index: int | None = None,
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_path, local_files_only=True
            )
            .to(device)
            .eval()
        )
        labels = self.model.config.id2label
        if entailment_index is None:
            matches = [
                int(k) for k, v in labels.items() if str(v).lower() == "entailment"
            ]
            if len(matches) != 1:
                raise ValueError(
                    "NLI config must identify entailment, or pass --entailment-index explicitly"
                )
            entailment_index = matches[0]
        if not 0 <= entailment_index < self.model.config.num_labels:
            raise ValueError("invalid entailment index")
        self.entailment_index = entailment_index
        self.device = device

    def score(self, statement: str, captions: list[str]) -> float:
        import torch

        if not captions:
            return 0.0
        inputs = self.tokenizer(
            captions,
            [statement] * len(captions),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**inputs).logits.float(), dim=-1)
        return float(probabilities[:, self.entailment_index].max().item())
