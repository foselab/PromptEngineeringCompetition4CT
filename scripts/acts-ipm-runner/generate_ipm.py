#!/usr/bin/env python3
"""Generate an ACTS input parameter model with an Ollama-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


PLACEHOLDER = "{SPECIFICATION}"
DEFAULT_LOCAL_HOST = "http://localhost:11434"
DEFAULT_CLOUD_HOST = "https://ollama.com"


class GenerationError(Exception):
    """A user-facing generation error."""


def read_utf8(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GenerationError(f"Could not read {label} '{path}': {exc}") from exc
    except UnicodeError as exc:
        raise GenerationError(f"{label.capitalize()} '{path}' is not valid UTF-8: {exc}") from exc


def choose_text(inline: str | None, file_path: Path | None, label: str) -> str:
    value = inline if inline is not None else read_utf8(file_path, label)  # type: ignore[arg-type]
    if not value.strip():
        raise GenerationError(f"The {label} must not be empty.")
    return value


def render_prompt(template: str, specification: str) -> str:
    if PLACEHOLDER not in template:
        raise GenerationError(
            f"The prompt must contain the literal placeholder {PLACEHOLDER}."
        )
    return template.replace(PLACEHOLDER, specification)


def api_endpoint(host: str) -> str:
    base = host.rstrip("/")
    if base.endswith("/api/generate"):
        return base
    if base.endswith("/api"):
        return base + "/generate"
    return base + "/api/generate"


def call_ollama(
    *,
    host: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
    temperature: float | None,
    seed: int | None,
    num_predict: int | None,
) -> tuple[str, dict[str, Any]]:
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        api_endpoint(host),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        try:
            detail = json.loads(detail).get("error", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise GenerationError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GenerationError(
            f"Could not connect to {api_endpoint(host)}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise GenerationError(f"The Ollama request timed out after {timeout:g} seconds.") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GenerationError("Ollama returned a response that was not valid JSON.") from exc
    if not isinstance(result, dict):
        raise GenerationError("Ollama returned an unexpected JSON response.")
    if result.get("error"):
        raise GenerationError(f"Ollama error: {result['error']}")
    generated = result.get("response")
    if not isinstance(generated, str) or not generated.strip():
        raise GenerationError("Ollama returned no generated text in the 'response' field.")
    return generated, result


def extract_acts(text: str) -> str:
    """Extract a fenced ACTS model when present; otherwise preserve the response."""
    fenced = re.findall(r"```(?:acts|txt|text)?\s*\n(.*?)```", text, flags=re.I | re.S)
    if fenced:
        selected = next((block for block in fenced if "[Parameter]" in block), fenced[0])
        return selected.strip() + "\n"
    return text.strip() + "\n"


def validate_acts(model: str) -> list[str]:
    errors: list[str] = []
    match = re.search(
        r"(?ims)^\s*\[Parameter\]\s*$\n(.*?)(?=^\s*\[[^\]]+\]\s*$|\Z)", model
    )
    if not match:
        return ["missing [Parameter] section"]
    definitions = [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.lstrip().startswith("--")
    ]
    if not definitions:
        errors.append("[Parameter] section is empty")
    elif not any(":" in line for line in definitions):
        errors.append("[Parameter] section contains no parameter definition")
    return errors


def safe_output_path(output_dir: Path, output_name: str) -> Path:
    name_path = Path(output_name)
    if name_path.name != output_name or output_name in {"", ".", ".."}:
        raise GenerationError("--output-name must be a file name, not a path.")
    if name_path.suffix.lower() != ".acts":
        output_name += ".acts"
    return output_dir / output_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an ACTS input parameter model from a natural-language specification."
    )
    parser.add_argument("--model", required=True, help="Ollama model name")

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 prompt template file")
    prompt_group.add_argument("--prompt", help="Prompt template supplied directly")

    spec_group = parser.add_mutually_exclusive_group(required=True)
    spec_group.add_argument(
        "--specification-file", type=Path, help="UTF-8 natural-language specification file"
    )
    spec_group.add_argument("--specification", help="Specification supplied directly")

    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory")
    parser.add_argument(
        "--output-name",
        help="Output file name (default: specification file stem, or ipm.acts)",
    )
    parser.add_argument(
        "--provider",
        choices=("local", "cloud"),
        default="local",
        help="Use local Ollama or the direct Ollama cloud API (default: local)",
    )
    parser.add_argument(
        "--host",
        help="Override the API host; OLLAMA_HOST is also honored in local mode",
    )
    parser.add_argument(
        "--api-key-env",
        default="OLLAMA_API_KEY",
        help="Environment variable containing the cloud API key (default: OLLAMA_API_KEY)",
    )
    parser.add_argument("--timeout", type=float, default=600, help="Request timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num-predict", type=int, default=8192, help="Maximum generated tokens"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.timeout <= 0:
            raise GenerationError("--timeout must be greater than zero.")
        if args.num_predict <= 0:
            raise GenerationError("--num-predict must be greater than zero.")

        template = choose_text(args.prompt, args.prompt_file, "prompt template")
        specification = choose_text(
            args.specification, args.specification_file, "specification"
        )
        prompt = render_prompt(template, specification)

        default_name = (
            f"{args.specification_file.stem}.acts"
            if args.specification_file is not None
            else "ipm.acts"
        )
        output_path = safe_output_path(args.output_dir, args.output_name or default_name)
        if output_path.exists() and not args.overwrite:
            raise GenerationError(
                f"Output file '{output_path}' already exists; use --overwrite to replace it."
            )

        if args.provider == "cloud":
            host = args.host or DEFAULT_CLOUD_HOST
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise GenerationError(
                    f"Cloud mode requires an API key in the {args.api_key_env} environment variable."
                )
        else:
            host = args.host or os.environ.get("OLLAMA_HOST") or DEFAULT_LOCAL_HOST
            # A signed-in local Ollama instance handles cloud-model authentication itself.
            # Do not leak a cloud key to a local or user-supplied host.
            api_key = None

        generated, _response = call_ollama(
            host=host,
            model=args.model,
            prompt=prompt,
            api_key=api_key,
            timeout=args.timeout,
            temperature=args.temperature,
            seed=args.seed,
            num_predict=args.num_predict,
        )
        acts_model = extract_acts(generated)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(acts_model)

        validation_errors = validate_acts(acts_model)
        print(output_path.resolve())
        if validation_errors:
            print(
                "Warning: generated output failed basic ACTS validation: "
                + "; ".join(validation_errors),
                file=sys.stderr,
            )
            return 2
        return 0
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
