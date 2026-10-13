"""Extraction backends (sentiment_proposal.md section 4.1).

A backend takes a system prompt, one user message and a JSON schema, and returns the
parsed JSON plus usage figures. `claude_code` runs Claude Code non-interactively on your
Pro subscription; `gemini` follows in S2 step 8.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Reply:
    payload: dict
    model: str  # the model that actually answered, e.g. claude-haiku-5-5
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)


class BackendError(Exception):
    """One call failed; worth one retry."""


class UsageLimitError(BackendError):
    """The subscription's usage limit is reached: stop the run, continue later."""


class Backend(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, schema: dict) -> Reply:
        ...


LIMIT_WORDS = ("usage limit", "rate limit", "limit reached", "limit will reset", "out of extra usage", "429")


class ClaudeCodeBackend:
    """`claude -p` with no tools, one turn and a JSON schema, from a neutral directory (so no
    project CLAUDE.md or memory is read); --system-prompt replaces Claude Code's own."""

    name = "claude_code"

    def __init__(self, model: str, timeout: float, executable: str | None = None) -> None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise BackendError("ANTHROPIC_API_KEY is set, so Claude Code would bill the API instead of your "
                               "subscription; unset it first")
        self.model, self.timeout = model, timeout
        self.executable = executable or shutil.which("claude")
        if not self.executable:
            raise BackendError("the `claude` command is not on PATH (install Claude Code and log in)")
        self.workdir = tempfile.mkdtemp(prefix="quant-extract-")

    def command(self, system: str, schema: dict) -> list[str]:
        # The prompt goes in a file: on Windows `claude` is a .CMD wrapper, and cmd.exe cannot pass
        # multi-line text with & or quotes safely, nor more than about 8,000 characters.
        prompt_file = os.path.join(self.workdir, f"system-{hashlib.sha256(system.encode()).hexdigest()[:16]}.txt")
        if not os.path.exists(prompt_file):
            with open(prompt_file, "w", encoding="utf-8") as target:
                target.write(system)
        return [self.executable, "-p", "--model", self.model, "--tools", "", "--strict-mcp-config",
                "--no-session-persistence", "--output-format", "json", "--system-prompt-file", prompt_file,
                "--json-schema", json.dumps(schema, separators=(",", ":"))]

    def complete(self, system: str, user: str, schema: dict) -> Reply:
        try:
            done = subprocess.run(self.command(system, schema), input=user, capture_output=True, text=True,
                                  encoding="utf-8", cwd=self.workdir, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"claude -p timed out after {self.timeout:g} s") from exc
        try:
            out = json.loads(done.stdout)
        except json.JSONDecodeError:
            out = None
        if out is None or done.returncode != 0 or out.get("is_error"):
            text = " ".join(filter(None, [done.stderr, str(out.get("result")) if out else done.stdout]))
            status = str(out.get("api_error_status")) if out else ""
            if status == "429" or any(word in text.lower() for word in LIMIT_WORDS):
                raise UsageLimitError(text.strip()[:300] or "usage limit")
            raise BackendError(f"claude -p failed (exit {done.returncode}): {text.strip()[:300]}")
        payload = out.get("structured_output")
        if not isinstance(payload, dict):
            try:
                payload = json.loads(out.get("result") or "")
            except json.JSONDecodeError as exc:
                raise BackendError("claude -p returned no JSON matching the schema") from exc
        usage = out.get("usage") or {}
        models = list(out.get("modelUsage") or {})
        return Reply(
            payload=payload, model=models[0] if models else self.model,
            input_tokens=int(usage.get("input_tokens", 0)) + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


def make_backend(name: str, model: str, timeout: float) -> Backend:
    if name == "claude_code":
        return ClaudeCodeBackend(model, timeout)
    raise BackendError(f"unknown extraction backend {name!r}")
