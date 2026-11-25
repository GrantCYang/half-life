"""Assistant-only LoRA training; heavy dependencies are optional and lazy."""

from __future__ import annotations
import json
import os
from pathlib import Path
from halflife.models import _load_model, render_candidate


def supervised_tokens(
    tokenizer, prompt: str, target: str, max_length: int = 1024
) -> dict:
    ids, prompt_length, _ = render_candidate(tokenizer, prompt, target)
    if len(ids) > max_length:
        raise ValueError(
            f"{len(ids)} tokens exceed max_length={max_length}; no truncation"
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * prompt_length + ids[prompt_length:],
    }


def gradient_accumulation(
    global_batch: int, world_size: int, microbatch: int = 1
) -> int:
    divisor = world_size * microbatch
    if min(global_batch, world_size, microbatch) <= 0 or global_batch % divisor:
        raise ValueError(
            "global batch must be a positive multiple of world_size * microbatch"
        )
    return global_batch // divisor


def train_lora(
    rows: list[dict],
    model_path: str,
    output_dir: str | Path,
    config: dict,
    *,
    device: str = "cuda",
    resume: str | None = None,
) -> dict:
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model

    if not rows:
        raise ValueError("training set is empty")
    if any(float(row["delta_sec"]) == 64.0 for row in rows):
        raise ValueError("64-second interpolation examples must not enter training")
    if any(row.get("split") not in {None, "train", "fit"} for row in rows):
        raise ValueError(
            "held-out/calibration/confirmation examples must not enter optimization"
        )
    if any(row.get("label") not in {"alive", "dead", 0, 1} for row in rows):
        raise ValueError("training accepts only labeled ALIVE/DEAD rows")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; choose an appropriate accelerator environment"
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif world_size > 1:
        raise ValueError(
            "this training entry point supports distributed CUDA execution only"
        )
    opt = config["optimization"]
    seed = opt["seed"]
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prepared = []
    for row in rows:
        target = "ALIVE" if row["label"] in ("alive", 1) else "DEAD"
        prepared.append(
            supervised_tokens(
                tokenizer, row["prompt"], target, config["max_sequence_length"]
            )
        )
    dtype = "float32" if device == "cpu" else config["dtype"]
    model = _load_model(
        model_path, device=device, dtype=dtype, attention=config["attn_implementation"]
    )
    lc = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc["rank"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.config.use_cache = False

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(prepared)

        def __getitem__(self, index):
            return prepared[index]

    def collate(features):
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if pad is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        maximum = max(len(f["input_ids"]) for f in features)
        fill = {"input_ids": pad, "attention_mask": 0, "labels": -100}
        return {
            key: torch.tensor(
                [f[key] + [fill[key]] * (maximum - len(f[key])) for f in features],
                dtype=torch.long,
            )
            for key in fill
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    accumulation = gradient_accumulation(
        opt["global_batch_size"], world_size, opt["microbatch_size"]
    )
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=opt["epochs"],
        per_device_train_batch_size=opt["microbatch_size"],
        gradient_accumulation_steps=accumulation,
        learning_rate=opt["learning_rate"],
        weight_decay=opt["weight_decay"],
        warmup_steps=opt["warmup_steps"],
        lr_scheduler_type=opt["scheduler"],
        max_grad_norm=opt["max_grad_norm"],
        seed=seed,
        data_seed=seed,
        bf16=dtype == "bfloat16",
        fp16=dtype == "float16",
        use_cpu=device == "cpu",
        logging_steps=5,
        save_steps=500,
        save_total_limit=2,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=Dataset(), data_collator=collate
    )
    result = trainer.train(resume_from_checkpoint=resume)
    receipt = {
        "examples": len(rows),
        "optimizer_steps": trainer.state.global_step,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "config": config,
        "metrics": result.metrics,
        "note": "Portable implementation; an execution receipt, not a claim to reproduce archived metrics.",
    }
    if trainer.is_world_process_zero():
        model.save_pretrained(out / "adapter")
        tokenizer.save_pretrained(out / "adapter")
        (out / "training_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    return receipt
