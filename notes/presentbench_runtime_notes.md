# PresentBench Runtime Notes

## Short Answer

PresentBench does not provide a slide-generation sandbox or generation agent.

It provides:

1. benchmark data: materials, instructions, judge prompts, weights;
2. evaluation code: load generated slides, run checklist judging, compute scores.

The slide generator is external. Our code must produce `slides.pdf` or `slides.pptx` under the expected result directory before PresentBench evaluation can run.

This is expected for our project. The few-shot examples are produced by our own teacher/refine loop, not supplied by PresentBench. PresentBench's role is to provide realistic heldout tasks, source materials, and checklist-based scoring.

## Official Flow

The official README flow is:

1. create Python 3.11 env and install requirements;
2. download dataset;
3. prepare slide agent outputs;
4. set Gemini-style API keys;
5. run `judge_all.py`.

The important directory contract is:

```text
<RESULT_ROOT>/<domain/.../case>/generation_task/results/
  slides.pdf
```

`slides.pptx` is accepted as fallback, but the evaluator converts it to PDF first.

## What Runs Locally

Local Python code handles:

- dataset traversal;
- finding each case's material files;
- finding generated `slides.pdf` or `slides.pptx`;
- assembling material-independent and material-dependent checklists;
- running some deterministic checks, such as slide count;
- aggregating yes/no checklist answers into weighted scores.

## What Uses External APIs

Most checklist items are judged by an external multimodal LLM API. The upstream CLI currently exposes:

- `gemini`
- `gemini_inline`

Although the repo contains an `OpenAIAPI` helper, `judge.py` currently only accepts `gemini` and `gemini_inline` in `--api_type`, so OpenAI is not a ready-to-use CLI path without patching.

## Sandbox / Docker

There is no Dockerfile and no sandbox requirement in the official repo.

Sandboxing is only relevant if our own slide-generation agent executes untrusted code, uses browser automation, or renders arbitrary HTML. PresentBench itself is just a Python evaluation pipeline plus external API calls.

## System Dependencies

Hard requirements for normal PDF-output evaluation:

- Python 3.11 environment recommended by upstream;
- Python packages from `requirements.txt`;
- `GENAI_API_KEY`, and optionally `GENAI_BASE_URL`;
- generated `slides.pdf` for each evaluated case.

Additional requirements if generated output is PPTX:

- LibreOffice CLI as `libreoffice` in `PATH`, because `utils/pptx_to_pdf.py` converts PPTX to PDF with LibreOffice headless mode.

Potential missing Python requirements not listed upstream:

- `Pillow`, because `utils/pptx_to_pdf.py` and `utils/pdf_to_images.py` import `PIL`;
- `python-pptx`, if using PPTX slide counting/truncation;
- `pdf2image` and Poppler `pdftoppm`, only for image-rendering helper paths.

## Local State Checked

On this machine:

- PresentBench data exists at `data/PresentBench_repo`;
- material LFS checkout is complete;
- Python 3.14 is installed globally, but upstream recommends Python 3.11;
- conda is available;
- Docker is not installed;
- LibreOffice is not installed;
- the global Python environment is missing most upstream Python dependencies.

## Recommendation

For our benchmark prototype:

1. Generate `slides.pdf` directly when possible. This avoids LibreOffice as a hard dependency.
2. Keep PresentBench evaluation code as a separate upstream checkout at `/Users/larry_1/Opensource/PresentBench`.
3. Use `auto-skill` to create result directories and invoke PresentBench evaluation.
4. Build the few-shot training examples ourselves from selected train tasks, then freeze only user-visible task/material/output content for induction.
5. Add a dedicated local env later, preferably `conda create -n presentbench python=3.11`, then install upstream requirements plus `Pillow` and `python-pptx`.
