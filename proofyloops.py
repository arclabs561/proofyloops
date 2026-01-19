from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from pydantic import BaseModel  # type: ignore[import-not-found]


def redact_secrets(text: str) -> str:
    """
    Best-effort redaction of common secret patterns.

    This is not a scanner. It's just a last-resort guard to reduce accidental leakage in review payloads.
    """
    # GitHub tokens
    text = re.sub(r"\bghp_[A-Za-z0-9]{20,}\b", "ghp_REDACTED", text)
    text = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "github_pat_REDACTED", text)
    # AWS access keys
    text = re.sub(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b", "AWS_ACCESS_KEY_REDACTED", text)
    # OpenAI-ish keys (very approximate; avoid overmatching short strings)
    text = re.sub(r"\bsk-[A-Za-z0-9]{20,}\b", "sk-REDACTED", text)
    return text


def _parse_dotenv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if k and k not in os.environ:
            out[k] = v
    return out


def load_dotenv_if_present(repo_root: Path) -> None:
    """
    Load `<repo_root>/.env` if present. Never override existing env vars.
    """
    p = repo_root / ".env"
    for k, v in _parse_dotenv(p).items():
        if os.environ.get(k) in (None, ""):
            os.environ[k] = v


def _find_lean_repo_root(start: Path) -> Path:
    """
    Walk upward until we see `lean-toolchain` and a lakefile.
    """
    cur = start.resolve()
    for _ in range(80):
        if (cur / "lean-toolchain").exists() and (
            (cur / "lakefile.lean").exists() or (cur / "lakefile.toml").exists()
        ):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(f"Could not find Lean repo root from {start}")


def _resolve_lake() -> Path:
    lake_env = os.environ.get("LAKE", "").strip()
    if lake_env:
        return Path(lake_env)
    elan_lake = Path.home() / ".elan" / "bin" / "lake"
    if elan_lake.exists():
        return elan_lake
    return Path("lake")


def _run_capture(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run a subprocess and capture stdout/stderr (utf-8, replacement on decode errors).
    """
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "timeout": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout.decode("utf-8", errors="replace"),
            "stderr": proc.stderr.decode("utf-8", errors="replace"),
            "cmd": list(cmd),
            "cwd": str(cwd),
        }
    except subprocess.TimeoutExpired as e:
        out_b = e.stdout or b""
        err_b = e.stderr or b""
        return {
            "ok": False,
            "timeout": True,
            "returncode": None,
            "stdout": out_b.decode("utf-8", errors="replace"),
            "stderr": err_b.decode("utf-8", errors="replace"),
            "cmd": list(cmd),
            "cwd": str(cwd),
        }


class ChatCompletionChoice(BaseModel):
    message: Dict[str, Any]


class ChatCompletionResponse(BaseModel):
    choices: List[ChatCompletionChoice]


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: Optional[str]
    model_env: str


def _providers() -> Dict[str, Provider]:
    return {
        "ollama": Provider(
            name="ollama",
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            api_key_env=None,
            model_env="OLLAMA_MODEL",
        ),
        "groq": Provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            model_env="GROQ_MODEL",
        ),
        "openai": Provider(
            name="openai",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key_env="OPENAI_API_KEY",
            model_env="OPENAI_MODEL",
        ),
        "openrouter": Provider(
            name="openrouter",
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            api_key_env="OPENROUTER_API_KEY",
            model_env="OPENROUTER_MODEL",
        ),
    }


def _provider_order() -> List[str]:
    raw = os.environ.get("PROOFYLOOPS_PROVIDER_ORDER", "").strip()
    if not raw:
        raw = os.environ.get("PROOFLOOPS_PROVIDER_ORDER", "").strip()
    if not raw:
        raw = os.environ.get("LEANPOT_PROVIDER_ORDER", "").strip()
    if not raw:
        return ["ollama", "groq", "openai", "openrouter"]
    return [x.strip() for x in raw.split(",") if x.strip()]


def _is_ollama_reachable(base_url: str, timeout_s: float) -> bool:
    candidates = [f"{base_url}/v1/models", f"{base_url}/api/tags"]
    try:
        with httpx.Client(timeout=timeout_s) as c:
            for u in candidates:
                r = c.get(u)
                if r.status_code in (200, 401, 404):
                    return True
    except Exception:
        return False
    return False


def _select_provider(timeout_s: float) -> Tuple[Provider, str]:
    provs = _providers()
    for name in _provider_order():
        p = provs.get(name)
        if p is None:
            continue
        model = os.environ.get(p.model_env, "").strip()
        if not model:
            continue
        if p.api_key_env:
            if not os.environ.get(p.api_key_env, "").strip():
                continue
        if p.name == "ollama":
            if not _is_ollama_reachable(p.base_url, timeout_s=timeout_s):
                continue
        return p, model
    raise RuntimeError(
        "No usable provider found. Set one of:\n"
        "- OLLAMA_MODEL (+ optional OLLAMA_HOST)\n"
        "- GROQ_API_KEY and GROQ_MODEL\n"
        "- OPENAI_API_KEY and OPENAI_MODEL\n"
        "- OPENROUTER_API_KEY and OPENROUTER_MODEL\n"
        "Optionally set PROOFYLOOPS_PROVIDER_ORDER."
    )


def chat_completion(*, system: str, user: str, timeout_s: float = 60.0) -> Dict[str, Any]:
    provider, model = _select_provider(timeout_s=3.0)

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if provider.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ.get(provider.api_key_env, '').strip()}"

    if provider.name == "openrouter":
        site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
        app_name = os.environ.get("OPENROUTER_APP_NAME", "").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }

    url = f"{provider.base_url}/chat/completions"
    with httpx.Client(timeout=timeout_s) as c:
        r = c.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    parsed = ChatCompletionResponse.model_validate(data)
    msg = parsed.choices[0].message
    content = msg.get("content", "")
    return {
        "provider": provider.name,
        "model": model,
        "content": content,
        "raw": data,
    }


def _extract_lemma_block(text: str, lemma_name: str) -> str:
    lines = text.splitlines()
    start = None
    pat = re.compile(rf"^\s*(theorem|lemma)\s+{re.escape(lemma_name)}\b")
    for i, ln in enumerate(lines):
        if pat.search(ln):
            start = i
            break
    if start is None:
        raise RuntimeError(f"Could not find theorem/lemma named {lemma_name}")

    end = None
    for j in range(start, min(len(lines), start + 250)):
        if ":=" in lines[j]:
            end = j
            break
    if end is None:
        end = min(len(lines) - 1, start + 80)
    tail_end = min(len(lines), end + 40)
    return "\n".join(lines[start:tail_end])


def _patch_first_sorry_in_lemma(*, text: str, lemma_name: str, replacement: str) -> Dict[str, Any]:
    """
    Replace the first line containing a standalone `sorry` token within the lemma block.

    The replacement is inserted with the same indentation as the `sorry` line.
    """
    lines = text.splitlines()
    start = None
    lemma_pat = re.compile(rf"^\s*(theorem|lemma)\s+{re.escape(lemma_name)}\b")
    for i, ln in enumerate(lines):
        if lemma_pat.search(ln):
            start = i
            break
    if start is None:
        raise RuntimeError(f"Could not find theorem/lemma named {lemma_name}")

    # Heuristic stop: within the next ~350 lines, or end-of-file.
    stop = min(len(lines), start + 350)

    sorry_line = None
    sorry_pat = re.compile(r"\bsorry\b")
    for j in range(start, stop):
        ln = lines[j]
        if ln.lstrip().startswith("--"):
            continue
        if sorry_pat.search(ln):
            sorry_line = j
            break

    if sorry_line is None:
        raise RuntimeError(f"Could not find a `sorry` token inside lemma {lemma_name}")

    indent = re.match(r"^\s*", lines[sorry_line]).group(0)  # type: ignore[union-attr]
    repl = replacement.rstrip("\n")
    if not repl.strip():
        raise RuntimeError("Empty replacement.")

    repl_lines = repl.splitlines()
    indented = [indent + r if r.strip() else r for r in repl_lines]

    before = lines[sorry_line]
    new_lines = lines[:sorry_line] + indented + lines[sorry_line + 1 :]
    after = "\n".join(indented)
    return {
        "text": "\n".join(new_lines) + ("\n" if text.endswith("\n") else ""),
        "changed": True,
        "line": sorry_line + 1,
        "indent": indent,
        "before": before,
        "after": after,
    }


def _lemma_block_contains_sorry(text: str, lemma_name: str) -> bool:
    block = _extract_lemma_block(text, lemma_name)
    return bool(re.search(r"\bsorry\b", block))


def verify_lean_text(*, repo_root: Path, lean_text: str, timeout_s: float = 120.0) -> Dict[str, Any]:
    """
    Run `lake env lean` on a temporary file containing `lean_text`.
    """
    repo_root = _find_lean_repo_root(repo_root)
    load_dotenv_if_present(repo_root)
    lake = _resolve_lake()

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".lean", delete=False) as f:
            f.write(lean_text)
            tmp_path = Path(f.name)
        cmd = [str(lake), "env", "lean", str(tmp_path)]
        res = _run_capture(cmd, cwd=repo_root, timeout_s=timeout_s)
        res["tmp_file"] = str(tmp_path)
        return res
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def verify_lean_file(*, repo_root: Path, file_rel: str, timeout_s: float = 120.0) -> Dict[str, Any]:
    repo_root = _find_lean_repo_root(repo_root)
    load_dotenv_if_present(repo_root)
    p = repo_root / file_rel
    if not p.exists():
        raise RuntimeError(f"File not found: {p}")
    txt = p.read_text(encoding="utf-8", errors="replace")
    return verify_lean_text(repo_root=repo_root, lean_text=txt, timeout_s=timeout_s)


def build_prompt_for_lemma(*, repo_root: Path, file_rel: str, lemma: str) -> Dict[str, Any]:
    """
    Build the (system,user) prompt pair used for proof suggestion, plus the extracted lemma excerpt.

    This is useful even without any configured LLM provider: it lets you run the prompt elsewhere and
    then feed the result back via `proofyloops patch`.
    """
    load_dotenv_if_present(repo_root)
    p = repo_root / file_rel
    if not p.exists():
        raise RuntimeError(f"File not found: {p}")
    txt = p.read_text(encoding="utf-8", errors="replace")
    block = _extract_lemma_block(txt, lemma)

    system = (
        "You are a Lean 4 proof assistant.\n"
        "Return ONLY Lean code that replaces a single `sorry` inside a `by` block.\n"
        "No markdown fences. No commentary. No surrounding `theorem`/`lemma` header.\n"
        "Prefer short tactic proofs (`simp`, `aesop`, `nlinarith`, `omega`, `ring_nf`) and existing lemmas.\n"
        "If you cannot complete the proof, return a minimal partial proof with the smallest remaining goal(s)."
    )
    user = (
        "We are working in a Lean 4 + Mathlib project.\n"
        "Here is the lemma context (excerpt):\n\n"
        f"{block}\n\n"
        "Task: provide the Lean proof code that replaces the `sorry` (the proof term only)."
    )

    return {
        "repo_root": str(repo_root),
        "file": str(p),
        "lemma": lemma,
        "excerpt": block,
        "system": system,
        "user": user,
    }


def suggest_proof_for_lemma(*, repo_root: Path, file_rel: str, lemma: str) -> Dict[str, Any]:
    prompt = build_prompt_for_lemma(repo_root=repo_root, file_rel=file_rel, lemma=lemma)
    res = chat_completion(system=prompt["system"], user=prompt["user"])
    return {
        "provider": res["provider"],
        "model": res["model"],
        "lemma": lemma,
        "file": prompt["file"],
        "suggestion": res["content"],
    }


def patch_and_verify(
    *,
    repo_root: Path,
    file_rel: str,
    lemma: str,
    replacement: str,
    timeout_s: float,
) -> Dict[str, Any]:
    repo_root = _find_lean_repo_root(repo_root)
    load_dotenv_if_present(repo_root)

    p = repo_root / file_rel
    if not p.exists():
        raise RuntimeError(f"File not found: {p}")
    original = p.read_text(encoding="utf-8", errors="replace")
    patched = _patch_first_sorry_in_lemma(text=original, lemma_name=lemma, replacement=replacement)
    still_has_sorry = _lemma_block_contains_sorry(patched["text"], lemma)
    verify = verify_lean_text(repo_root=repo_root, lean_text=patched["text"], timeout_s=timeout_s)
    return {
        "file": str(p),
        "lemma": lemma,
        "patch": {k: patched[k] for k in ("line", "before", "after", "indent")},
        "lemma_still_contains_sorry": still_has_sorry,
        "verify": verify,
    }


def loop_once(
    *,
    repo_root: Path,
    file_rel: str,
    lemma: str,
    timeout_s: float,
) -> Dict[str, Any]:
    repo_root = _find_lean_repo_root(repo_root)
    load_dotenv_if_present(repo_root)

    p = repo_root / file_rel
    if not p.exists():
        raise RuntimeError(f"File not found: {p}")
    original = p.read_text(encoding="utf-8", errors="replace")

    suggestion = suggest_proof_for_lemma(repo_root=repo_root, file_rel=file_rel, lemma=lemma)
    patched = _patch_first_sorry_in_lemma(text=original, lemma_name=lemma, replacement=suggestion["suggestion"])

    still_has_sorry = _lemma_block_contains_sorry(patched["text"], lemma)
    verify = verify_lean_text(repo_root=repo_root, lean_text=patched["text"], timeout_s=timeout_s)

    return {
        "suggestion": suggestion,
        "patch": {k: patched[k] for k in ("line", "before", "after", "indent")},
        "lemma_still_contains_sorry": still_has_sorry,
        "verify": verify,
    }


def loop_repair(
    *,
    repo_root: Path,
    file_rel: str,
    lemma: str,
    max_iters: int,
    timeout_s: float,
) -> Dict[str, Any]:
    """
    Bounded suggest→patch→verify loop over a single file.

    This does NOT write back to the repo. It returns a trace of attempts.
    """
    if max_iters <= 0:
        raise RuntimeError("max_iters must be >= 1")

    repo_root = _find_lean_repo_root(repo_root)
    load_dotenv_if_present(repo_root)

    p = repo_root / file_rel
    if not p.exists():
        raise RuntimeError(f"File not found: {p}")
    cur_text = p.read_text(encoding="utf-8", errors="replace")

    attempts: List[Dict[str, Any]] = []
    for i in range(max_iters):
        suggestion = suggest_proof_for_lemma(repo_root=repo_root, file_rel=file_rel, lemma=lemma)

        patched = _patch_first_sorry_in_lemma(text=cur_text, lemma_name=lemma, replacement=suggestion["suggestion"])
        if patched["text"] == cur_text:
            attempts.append(
                {
                    "iter": i,
                    "suggestion": suggestion,
                    "patch": {k: patched[k] for k in ("line", "before", "after", "indent")},
                    "note": "no_change",
                }
            )
            break

        cur_text = patched["text"]
        still_has_sorry = _lemma_block_contains_sorry(cur_text, lemma)
        verify = verify_lean_text(repo_root=repo_root, lean_text=cur_text, timeout_s=timeout_s)

        attempts.append(
            {
                "iter": i,
                "suggestion": suggestion,
                "patch": {k: patched[k] for k in ("line", "before", "after", "indent")},
                "lemma_still_contains_sorry": still_has_sorry,
                "verify": verify,
            }
        )

        # Stop if (a) lemma has no `sorry` and (b) Lean elaborates.
        if (not still_has_sorry) and bool(verify.get("ok")):
            break

    return {
        "repo_root": str(repo_root),
        "file": str(p),
        "lemma": lemma,
        "max_iters": max_iters,
        "attempts": attempts,
        "final_lemma_contains_sorry": _lemma_block_contains_sorry(cur_text, lemma),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="proofyloops", description="Local Lean proof helper.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suggest", help="Suggest a proof for a lemma in a file.")
    s.add_argument("--repo", required=True, help="Repo root (directory containing lean-toolchain).")
    s.add_argument("--file", required=True, help="Lean file path relative to repo root.")
    s.add_argument("--lemma", required=True, help="Lemma/theorem name to target.")

    pr = sub.add_parser("prompt", help="Print the prompt payload (no LLM call).")
    pr.add_argument("--repo", required=True, help="Repo root (directory containing lean-toolchain).")
    pr.add_argument("--file", required=True, help="Lean file path relative to repo root.")
    pr.add_argument("--lemma", required=True, help="Lemma/theorem name to target.")

    pa = sub.add_parser("patch", help="Apply a replacement (in-memory) and verify with Lean.")
    pa.add_argument("--repo", required=True, help="Repo root (directory containing lean-toolchain).")
    pa.add_argument("--file", required=True, help="Lean file path relative to repo root.")
    pa.add_argument("--lemma", required=True, help="Lemma/theorem name to target.")
    pa.add_argument(
        "--replacement-file",
        required=True,
        help="Path to a file containing the Lean proof code to splice in (replaces the first `sorry`).",
    )
    pa.add_argument("--timeout-s", type=float, default=120.0, help="Timeout seconds for Lean.")

    v = sub.add_parser("verify", help="Run `lake env lean` on a file (elaboration check).")
    v.add_argument("--repo", required=True, help="Repo root (directory containing lean-toolchain).")
    v.add_argument("--file", required=True, help="Lean file path relative to repo root.")
    v.add_argument("--timeout-s", type=float, default=120.0, help="Timeout seconds for Lean.")

    l = sub.add_parser("loop", help="Bounded suggest→patch→verify loop for one lemma.")
    l.add_argument("--repo", required=True, help="Repo root (directory containing lean-toolchain).")
    l.add_argument("--file", required=True, help="Lean file path relative to repo root.")
    l.add_argument("--lemma", required=True, help="Lemma/theorem name to target.")
    l.add_argument("--timeout-s", type=float, default=120.0, help="Timeout seconds for Lean.")
    l.add_argument("--max-iters", type=int, default=3, help="Max iterations (bounded).")

    r = sub.add_parser("review-diff", help="LLM review of repo git diff (project-agnostic).")
    r.add_argument("--repo", required=True, help="Repo root (directory containing lean-toolchain).")
    r.add_argument(
        "--scope",
        default="staged",
        choices=["staged", "unstaged"],
        help="Which diff to review (default: staged).",
    )
    r.add_argument("--max-diff-bytes", type=int, default=180_000)
    r.add_argument("--max-file-bytes", type=int, default=12_000)
    r.add_argument("--max-total-bytes", type=int, default=320_000)
    r.add_argument(
        "--prompt-only",
        action="store_true",
        help="Do not call an LLM; emit the bounded prompt JSON and exit 0.",
    )
    r.add_argument(
        "--require-key",
        action="store_true",
        help="Fail if no provider is configured (otherwise prints a skip message and exits 0).",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "suggest":
        repo_root = Path(args.repo).expanduser().resolve()
        out = suggest_proof_for_lemma(repo_root=repo_root, file_rel=args.file, lemma=args.lemma)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    if args.cmd == "prompt":
        repo_root = Path(args.repo).expanduser().resolve()
        out = build_prompt_for_lemma(repo_root=repo_root, file_rel=args.file, lemma=args.lemma)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    if args.cmd == "patch":
        repo_root = Path(args.repo).expanduser().resolve()
        repl_path = Path(args.replacement_file).expanduser().resolve()
        replacement = repl_path.read_text(encoding="utf-8", errors="replace")
        out = patch_and_verify(
            repo_root=repo_root,
            file_rel=args.file,
            lemma=args.lemma,
            replacement=replacement,
            timeout_s=float(args.timeout_s),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    if args.cmd == "verify":
        repo_root = Path(args.repo).expanduser().resolve()
        out = verify_lean_file(repo_root=repo_root, file_rel=args.file, timeout_s=float(args.timeout_s))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    if args.cmd == "loop":
        repo_root = Path(args.repo).expanduser().resolve()
        out = loop_repair(
            repo_root=repo_root,
            file_rel=args.file,
            lemma=args.lemma,
            max_iters=int(args.max_iters),
            timeout_s=float(args.timeout_s),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "review-diff":
        repo_root = Path(args.repo).expanduser().resolve()
        load_dotenv_if_present(repo_root)

        def _git(cmd: Sequence[str]) -> str:
            proc = subprocess.run(list(cmd), cwd=str(repo_root), capture_output=True)
            out = proc.stdout.decode("utf-8", errors="replace")
            err = proc.stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(f"git failed: {cmd}\n{err}")
            return out

        if args.scope == "staged":
            diff_cmd = ["git", "diff", "--cached"]
            names_cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
        else:
            diff_cmd = ["git", "diff"]
            names_cmd = ["git", "diff", "--name-only", "--diff-filter=ACMR"]

        diff = _git(diff_cmd)
        if len(diff.encode("utf-8")) > int(args.max_diff_bytes):
            b = diff.encode("utf-8")[: int(args.max_diff_bytes)]
            diff = b.decode("utf-8", errors="replace") + "\n…(truncated)\n"
        diff = redact_secrets(diff)

        changed = [ln.strip() for ln in _git(names_cmd).splitlines() if ln.strip()]

        def _is_sensitive_path(rel: str) -> bool:
            # Conservative: do not include common secret locations, VCS internals, or binary-ish artifacts.
            rel_norm = rel.replace("\\", "/")
            parts = [p for p in rel_norm.split("/") if p]
            if not parts:
                return True

            if any(p in {".git", ".venv", "node_modules", "target", ".cursor"} for p in parts):
                return True

            base = parts[-1].lower()
            if base in {
                ".env",
                ".envrc",
                ".npmrc",
                ".pypirc",
                "credentials",
                "credentials.json",
                "id_rsa",
                "id_ed25519",
                "known_hosts",
            }:
                return True

            bad_exts = {
                ".pem",
                ".key",
                ".p12",
                ".pfx",
                ".kdbx",
                ".sqlite",
                ".db",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".pdf",
                ".zip",
                ".tar",
                ".gz",
                ".bz2",
                ".xz",
            }
            return any(base.endswith(ext) for ext in bad_exts)

        # Collect small excerpts for changed files (best-effort).
        total_used = 0
        excerpts: List[Dict[str, Any]] = []
        for rel in changed:
            pth = (repo_root / rel).resolve()
            if _is_sensitive_path(rel):
                continue
            try:
                pth.relative_to(repo_root)
            except Exception:
                continue
            if not pth.is_file():
                continue
            try:
                raw = pth.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            b = raw.encode("utf-8", errors="replace")
            if total_used >= int(args.max_total_bytes):
                break
            take = min(len(b), int(args.max_file_bytes), int(args.max_total_bytes) - total_used)
            total_used += take
            s = b[:take].decode("utf-8", errors="replace")
            if take < len(b):
                s += "\n…(truncated)\n"
            excerpts.append({"path": rel, "bytes": take, "text": redact_secrets(s)})

        system = (
            "You are a skeptical code reviewer for a Lean/mathlib-focused repository.\n"
            "Priorities:\n"
            "- correctness and proof soundness\n"
            "- API stability / portability (Linux case sensitivity, CI)\n"
            "- minimal diffs (prefer small fixes)\n"
            "Avoid praise. If unsure, say what is missing.\n"
            "Return a concise review with concrete, actionable fixes.\n"
        )

        user = json.dumps(
            {
                "repo_root": str(repo_root),
                "scope": args.scope,
                "changed_paths": changed,
                "diff": diff,
                "excerpts": excerpts,
            },
            ensure_ascii=False,
        )

        if args.prompt_only:
            print(
                json.dumps(
                    {
                        "repo_root": str(repo_root),
                        "scope": args.scope,
                        "changed_paths": changed,
                        "diff": diff,
                        "excerpts": excerpts,
                        "system": system,
                        "user": user,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        try:
            resp = chat_completion(system=system, user=user, timeout_s=90.0)
        except Exception as e:
            if args.require_key:
                raise
            print(f"proofyloops review-diff: no provider configured or request failed; skipping ({e})", file=sys.stderr)
            return

        out = {
            "repo_root": str(repo_root),
            "scope": args.scope,
            "provider": resp.get("provider"),
            "model": resp.get("model"),
            "review": resp.get("content", ""),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

