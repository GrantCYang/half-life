# HALF-LIFE

Whether a remembered statement still holds can be learned from automatically labeled video. On volatile statements at 120 seconds, the adapted forecaster obtains Brier **0.112**, compared with **0.126** for contextual ridge.

[Paper](paper/main.pdf) · [Source](paper/main.tex) · [Data](data/README.md) · [Downloads](https://github.com/GrantCYang/half-life/releases/tag/v0.2.1)

The manuscript is available here as a preprint. It has not yet been posted to arXiv; a permanent article link will be added when available.

## CPU check

With Python 3.10 or later, from the repository root:

```sh
python -m pip install -e ".[figures]"
python scripts/replay.py
```

This validates the published aggregate table relationships and regenerates the Brier and fixed-rate recall figures. It does not recompute take-level uncertainty or the private 216-value forecasting and 4,320-value captioning replays. Per-row inputs, predictions, captions and grouping records are not included.

## Method

`src/halflife/data` implements past-only memory construction and automatic supervision. Later captions can confirm validity; a visual interrogator labels remaining examples still true, no longer true or undeterminable. Unknown targets stay masked. Future evidence is used only for labeling.

`src/halflife/forecast` contains baseline fitting, LoRA training, likelihood scoring, calibration and metrics. `src/halflife/caption` prepares equal-budget memory selections and caption requests. Use `halflife --help` for the file-oriented pipeline and `halflife baselines --help` for B0-B4 fitting; model commands require authorized local checkpoints and data supplied separately.

Forecast, caption and earlier-implementation aggregates live in separate directories under `data/`. Caption-selection results show suggestive gains over contextual ridge: most automatic-evaluation point estimates and the human-reference-assisted estimate favor the forecaster, but their confidence intervals include zero. These comparisons share samples and are not independent replications. Outcome-specific Brier and fixed-rate recall answer different questions.

## Models and data

The recorded base is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It uses the Qwen3.5 architecture. Model metadata and shard identities are in `configs/`. Weights and adapters are not redistributed. `configs/label_model_identity.json` identifies the preserved NLI and sentence-encoder repositories and weight-file hashes. Their upstream commit revisions and complete historical run binding remain unresolved; full model regeneration has not been validated in a relocated environment.

Obtain source videos through [Ego-Exo4D](https://ego-exo4d-data.org/). The two frames embedded in the paper are attributed publication illustrations, not a redistributed dataset. See [third-party terms](THIRD_PARTY.md).

## Paper and citation

Build the preprint from `paper/` with PDFLaTeX and BibTeX:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The source package prepared for arXiv includes the bibliography, styles and figures, with companion code and aggregate data under `anc/`. It is available in the release downloads. This release is version **0.2.1** and updates presentation and documentation; experimental results are unchanged. The author is Yixuan Yang, independent researcher. Citation metadata is in `CITATION.cff`.

The public scope is method code, synthetic workflow examples and recorded aggregate results. Statistical recomputation from individual predictions, retraining and new video/model inference require inputs not distributed here.

Project code is MIT licensed. Original aggregate tables are CC BY 4.0; third-party media and styles retain their own terms.
