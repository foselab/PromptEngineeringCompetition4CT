# ACTS IPM generator

`generate_ipm.py` inserts a natural-language specification into a participant's prompt, sends the result to an Ollama model, and saves the generated input parameter model as an `.acts` file.

It uses only the Python standard library and works with Python 3.9 or newer.

## Files supplied by a participant

The prompt template must contain the literal placeholder `{SPECIFICATION}`. For example:

```text
Create an ACTS input parameter model for the specification below.
Return only valid ACTS content, beginning with [System], with no Markdown fences or explanation.

Specification:
{SPECIFICATION}
```

The specification is an ordinary UTF-8 text file.

## Local Ollama

Start Ollama, make sure the model is available, and run:

```bash
python3 generate_ipm.py \
  --provider local \
  --model qwen3:8b \
  --prompt-file participant-prompt.txt \
  --specification-file specification.txt \
  --output-dir results
```

The default local endpoint is `http://localhost:11434`. Set `OLLAMA_HOST` or pass `--host` to use another Ollama-compatible endpoint.

Cloud-suffixed models can also be reached through a signed-in local Ollama instance; in that case, keep `--provider local` and use a model such as `gpt-oss:120b-cloud`.

## Direct Ollama cloud API

Create an Ollama API key, place it in the environment, and select the cloud provider:

```bash
export OLLAMA_API_KEY="your-key"
python3 generate_ipm.py \
  --provider cloud \
  --model gpt-oss:120b \
  --prompt-file participant-prompt.txt \
  --specification-file specification.txt \
  --output-dir results
```

Direct cloud mode uses `https://ollama.com/api/generate`. The key is never accepted as a command-line value, which keeps it out of shell history and process listings.

## Output and exit status

By default, a specification named `checkout.txt` produces `results/checkout.acts`. Use `--output-name team-7.acts` to choose a different name and `--overwrite` to replace an existing file.

The script removes a surrounding Markdown code fence if the model adds one. It performs a deliberately small format check: the output must have a non-empty `[Parameter]` section containing a parameter definition. Full syntactic and semantic validation should be done by running the generated file through the competition's ACTS version.

Exit codes:

- `0`: generation and basic validation succeeded
- `1`: arguments, file access, connection, API, or writing failed
- `2`: output was saved, but failed basic ACTS validation

Run `python3 generate_ipm.py --help` for all options. Generation defaults to temperature `0`, seed `0`, and at most `8192` generated tokens; each can be overridden.

## Inline values

For quick checks, files are optional:

```bash
python3 generate_ipm.py \
  --model gemma3 \
  --prompt 'Generate an ACTS model for: {SPECIFICATION}' \
  --specification 'A form has a browser choice and a logged-in flag.' \
  --output-dir results \
  --output-name form.acts
```
