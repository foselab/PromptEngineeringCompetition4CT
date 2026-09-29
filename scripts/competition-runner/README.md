# Competition runner

`competition_runner.py` executes a participant prompt on every textual SUT specification, saves each generated ACTS IPM, compares it with the corresponding reference IPM, and produces detailed JSON and text reports.

The runner reuses:

- `../acts-ipm-runner/generate_ipm.py` for prompt substitution and Ollama generation;
- `../acts-ipm-equivalence-checker/acts_equiv.py` for ACTS parsing and SMT-based constraint equivalence.

## Setup

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r scripts/competition-runner/requirements.txt
```

For the default semantic name evaluation, make the embedding model available to the same Ollama instance:

```bash
ollama pull embeddinggemma
```

## Development run with one LLM

```bash
python3 scripts/competition-runner/competition_runner.py \
  --specifications-dir benchmarks/development/specifications \
  --references-dir benchmarks/development/IPM \
  --prompt-file participant-prompt.txt \
  --model qwen3:8b \
  --output-dir results/development
```

Specification and reference files are paired by stem. For example, `checkout.txt` is matched with `checkout.acts`, or with `checkout.txt` if no `.acts` reference exists.

## Final run with known and transfer LLMs

Repeat `--model` to implement the proposal's final-score rule:

```bash
python3 scripts/competition-runner/competition_runner.py \
  --specifications-dir benchmarks/evaluation/specifications \
  --references-dir benchmarks/evaluation/IPM \
  --prompt-file participant-prompt.txt \
  --model known-model:tag \
  --model transfer-model:tag \
  --output-dir results/final
```

Each benchmark receives the raw weighted score defined by the proposal. A model's score is the sum of its benchmark scores, and the final score is the sum across all models. Because the raw maximum depends on the number of parameters in each reference IPM, the report always displays the achieved and maximum raw scores together. It also includes normalized 0–100 benchmark, model, and final scores for easier interpretation; these do not replace the raw final score used by the formula.

Use `--provider cloud` with `OLLAMA_API_KEY` for direct Ollama Cloud access. `--host` and `--embedding-host` can override the generation and embedding endpoints.

## Scoring

The implementation makes the proposal's candidate formula executable and records all choices in `report.json`:

- Parameter and level count: `max(0, 1 - abs(reference - predicted) / reference)`.
- Type correctness: exact, binary comparison for aligned parameters.
- Parameter and level names: exact matches score 1; otherwise embeddings are compared with cosine similarity. Similarities at or below the cutoff score 0, and values above it are linearly scaled to 1.
- Constraints: binary semantic equivalence of the complete valid-configuration spaces, evaluated by Z3. Constraint text and count do not affect this component.
- Default weights: `w1=5`, `w2=4`, `w3=3`, `w4=2`, `w5=1`, satisfying `w1 > w2 > w3 > w4 > w5`.

The raw weighted score is also normalized to 0–100 for reporting. Override weights with `--w1` through `--w5`; the strict ordering is enforced.

Use `--name-similarity exact` to run without an embedding model. This diagnostic mode does not implement the proposal's semantic-name credit and should not be used for the official ranking.

## Output

The output directory contains:

```text
generated/
  01-model-name/
    benchmark-name.acts
report.json
report.txt
```

The reports include the final and per-model scores, an aggregate error summary, every benchmark's component scores, semantic alignments, missing and spurious parameters or values, type mismatches, invalid syntax or sections, generation failures, and constraint counterexamples when available.

Run with `--overwrite` to replace existing reports.
