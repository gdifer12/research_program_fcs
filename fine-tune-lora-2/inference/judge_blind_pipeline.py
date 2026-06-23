#!/usr/bin/env python3
"""
Blind judge pipeline for generated samples.

Two main operations:

1) prepare / pack / anonymize
   Reads several model generation files from a run directory, removes model-identifying
   fields, shuffles all samples together, and writes one anonymous JSONL file for a judge.
   A private map JSON is also written; do NOT send it to the judge.

2) restore / unpack / deanonymize
   Reads the judge JSONL output and the private map, then restores judged samples back
   into per-source-file JSONL files.

Expected judge-visible schema:
{
  "sample_id": "string",
  "prompt_id": "string",
  "prompt_type": "continuation | instruct | constraints | ood_simple",
  "prompt": "string",
  "generated_text": "string",
  "max_new_tokens": 128
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# =======================================================
# Generic helpers
# =======================================================

ALLOWED_PROMPT_TYPES = {"continuation", "instruct", "constraints", "ood_simple"}

JUDGE_FIELDS = [
    "sample_id",
    "prompt_id",
    "prompt_type",
    "prompt",
    "generated_text",
    "max_new_tokens",
]

SCORE_FIELDS = [
    "english_fluency",
    "coherence",
    "task_fulfillment",
    "constraint_following",
    "degeneration_control",
    "completeness",
    "overall_quality",
]

FLAG_FIELDS = [
    "empty_or_near_empty",
    "mostly_non_english",
    "mostly_incoherent",
    "major_prompt_mismatch",
    "explicit_constraint_violation",
    "severe_repetition_or_looping",
    "likely_truncated",
]

# fields that must never be sent to the blind judge
HIDDEN_FIELD_RE = re.compile(
    r"(" 
    r"model|hash|checkpoint|ckpt|path|seed|temperature|top_p|top_k|" 
    r"lora|qlora|quant|loss|perplexity|ppl|run|commit|config|" 
    r"expected|tag|note|behavior|judge|score|flag"
    r")",
    re.IGNORECASE,
)

PATH_KEY_RE = re.compile(r"(file|path|json|jsonl|output|generation|sample|result)", re.IGNORECASE)
JSON_SUFFIXES = {".json", ".jsonl"}


def ensure_dir(path: str | Path, die: bool = True) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        if die:
            raise FileNotFoundError(f"No such directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_file(path: str | Path, die: bool = True) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_file():
        if die:
            raise FileNotFoundError(f"No such file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return path


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def sanitize_filename(s: str, default: str = "unknown") -> str:
    s = str(s or "").strip()
    if not s:
        s = default
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s)
    s = s.strip("._-")
    return s or default



# These are names of meta fields, not meaningful model/source names.
# They must not become restored output groups like judged_output_file.jsonl.
GENERIC_SOURCE_KEYS = {
    "file", "path", "json", "jsonl", "output", "output_file",
    "generation", "generation_file", "sample", "sample_file",
    "result", "result_file", "source_file", "model_file",
}

def is_generic_source_key(value: Any) -> bool:
    key = sanitize_filename(str(value or "").strip(), default="").lower()
    return (not key) or key in GENERIC_SOURCE_KEYS

def effective_source_key(
    source_key: Any = None,
    source_stem: Any = None,
    source_file: Any = None,
    default: str = "unknown",
) -> str:
    """Return a stable non-generic grouping key for restored files."""
    key = sanitize_filename(source_key or "", default="")
    stem = sanitize_filename(source_stem or "", default="")

    if not stem and source_file:
        try:
            stem = sanitize_filename(Path(str(source_file)).stem, default="")
        except Exception:
            stem = ""

    if is_generic_source_key(key):
        return stem or default
    return key or stem or default


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    return x


def load_json(path: str | Path) -> Any:
    path = ensure_file(path, die=True)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_code_fence(text: str) -> str:
    """Accept judge answers accidentally wrapped into ```jsonl ... ``` fences."""
    t = text.strip()
    if not t.startswith("```"):
        return text

    lines = t.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"


def extract_list_from_json(data: Any, path: Path) -> list[dict[str, Any]]:
    """
    Accepts:
      - JSON list of objects
      - JSON object with a common list field: samples/generations/results/data/items/records
      - single JSON object representing one sample
    """
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
        for key in ("samples", "generations", "results", "data", "items", "records", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None:
            rows = [data]
    else:
        raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data).__name__}")

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Non-object row #{i} in {path}")
        out.append(row)
    return out


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Reads JSONL or JSON. The extension is ignored because some files named .json
    are actually JSONL in this pipeline.
    """
    path = ensure_file(path, die=True)
    text = path.read_text(encoding="utf-8")
    text = strip_code_fence(text)
    if not text.strip():
        return []

    # First try normal JSON. If this fails with extra data, treat it as JSONL.
    try:
        data = json.loads(text)
        return extract_list_from_json(data, path)
    except json.JSONDecodeError:
        pass

    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Bad JSONL in {path}:{lineno}: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL row is not an object in {path}:{lineno}")
        rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cnt = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            cnt += 1
    return cnt


def get_first(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


# =======================================================
# Input discovery
# =======================================================

@dataclass
class SourceFile:
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return effective_source_key(self.meta.get("source_key"), source_stem=self.path.stem, source_file=self.path)


def resolve_candidate_path(value: str, run_dir: Path) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    # Strip common URI-ish prefixes and quotes.
    raw = raw.strip("'\"")
    if raw.startswith("file://"):
        raw = raw[len("file://") :]

    p = Path(raw).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(run_dir / p)
        candidates.append(Path.cwd() / p)
        candidates.append(p)

    # Sometimes meta stores only basename.
    if p.name == raw:
        candidates.append(run_dir / p.name)

    for c in candidates:
        try:
            if c.is_file() and c.suffix.lower() in JSON_SUFFIXES:
                return c.resolve()
        except OSError:
            continue
    return None


def pick_hidden_meta(obj: dict[str, Any]) -> dict[str, Any]:
    """Fields useful for restoring/grouping, but never sent to the judge."""
    keep = {}
    for k, v in obj.items():
        ks = str(k)
        if HIDDEN_FIELD_RE.search(ks):
            # Avoid storing huge nested data accidentally from meta.json.
            if isinstance(v, (str, int, float, bool)) or v is None:
                keep[ks] = v
            elif isinstance(v, Path):
                keep[ks] = str(v)
    return keep


def find_source_files_from_meta(meta: dict[str, Any], run_dir: Path) -> list[SourceFile]:
    found: dict[Path, SourceFile] = {}

    def add(path: Path, obj_meta: dict[str, Any] | None = None) -> None:
        if path.name == "meta.json":
            return
        sf = found.get(path)
        if sf is None:
            sf = SourceFile(path=path, meta={})
            found[path] = sf
        if obj_meta:
            sf.meta.update({k: v for k, v in obj_meta.items() if v is not None})

    def walk(x: Any, context: dict[str, Any] | None = None) -> None:
        context = context or {}
        if isinstance(x, dict):
            local_hidden = pick_hidden_meta(x)
            local_context = {**context, **local_hidden}

            # Direct path-like values in this object.
            for k, v in x.items():
                if isinstance(v, str) and PATH_KEY_RE.search(str(k)):
                    p = resolve_candidate_path(v, run_dir)
                    if p is not None:
                        meta_for_file = dict(local_context)
                        meta_for_file.setdefault("source_key", x.get("name") or x.get("model") or x.get("model_name") or Path(v).stem)
                        add(p, meta_for_file)

            # Dicts like {"ftl_F_1": "ftl_F_1.json"}.
            for k, v in x.items():
                if isinstance(v, str):
                    p = resolve_candidate_path(v, run_dir)
                    if p is not None:
                        meta_for_file = dict(local_context)
                        if str(k).lower() in {"file", "path", "json", "jsonl", "output", "generation", "sample", "result"}:
                            meta_for_file.setdefault("source_key", x.get("name") or x.get("model") or x.get("model_name") or p.stem)
                        else:
                            meta_for_file.setdefault("source_key", str(k) if not str(k).startswith("/") else p.stem)
                        add(p, meta_for_file)

            for v in x.values():
                walk(v, local_context)

        elif isinstance(x, list):
            for v in x:
                walk(v, context)
        elif isinstance(x, str):
            p = resolve_candidate_path(x, run_dir)
            if p is not None:
                add(p, context)

    walk(meta)
    return sorted(found.values(), key=lambda sf: sf.path.name)


def find_source_files_by_scan(run_dir: Path, excluded: set[Path]) -> list[SourceFile]:
    files: list[SourceFile] = []
    for p in sorted(run_dir.iterdir()):
        if not p.is_file():
            continue
        if p.resolve() in excluded:
            continue
        if p.suffix.lower() not in JSON_SUFFIXES:
            continue
        if p.name == "meta.json":
            continue
        if any(bad in p.stem.lower() for bad in ("blind", "judge", "judged", "map", "restore_report")):
            continue
        files.append(SourceFile(path=p.resolve(), meta={"source_key": p.stem}))
    return files


def discover_source_files(
    run_dir: Path,
    meta_path: Path | None,
    input_files: list[str] | None,
) -> tuple[list[SourceFile], dict[str, Any]]:
    meta: dict[str, Any] = {}
    found: dict[Path, SourceFile] = {}

    if meta_path is not None and meta_path.is_file():
        loaded = load_json(meta_path)
        if isinstance(loaded, dict):
            meta = loaded
            for sf in find_source_files_from_meta(meta, run_dir):
                found[sf.path.resolve()] = sf
        else:
            raise ValueError(f"meta must be a JSON object: {meta_path}")

    if input_files:
        for i, raw in enumerate(input_files):
            p = resolve_candidate_path(raw, run_dir)
            if p is None:
                raise FileNotFoundError(f"Input file not found: {raw}")
            found[p.resolve()] = SourceFile(path=p.resolve(), meta={"source_key": p.stem, "manual_index": i})

    # Fallback: exactly the screenshot-like case, all generation files are in runs/E_1.
    if not found:
        excluded = set()
        if meta_path is not None:
            excluded.add(meta_path.resolve())
        for sf in find_source_files_by_scan(run_dir, excluded):
            found[sf.path.resolve()] = sf

    files = sorted(found.values(), key=lambda sf: (sf.meta.get("manual_index", 10**9), sf.path.name))
    return files, meta


# =======================================================
# Preparing blind input
# =======================================================

@dataclass
class PackedSample:
    blind: dict[str, Any]
    entry: dict[str, Any]


def normalize_for_judge(raw: dict[str, Any], source: SourceFile, row_index: int) -> PackedSample:
    prompt_id = get_first(raw, ("prompt_id", "id", "promptId", "prompt_idx", "prompt_index"))
    prompt_type = get_first(raw, ("prompt_type", "type", "category"))
    prompt = get_first(raw, ("prompt", "input", "instruction"))
    generated_text = get_first(raw, ("generated_text", "completion", "output", "response", "answer", "text"))
    max_new_tokens = get_first(raw, ("max_new_tokens", "max_tokens", "new_tokens", "generation_max_new_tokens"))

    missing = []
    if prompt_id is None:
        missing.append("prompt_id/id")
    if prompt_type is None:
        missing.append("prompt_type/type")
    if prompt is None:
        missing.append("prompt")
    if generated_text is None:
        missing.append("generated_text/output/completion/text")
    if missing:
        raise ValueError(
            f"Cannot normalize row #{row_index} from {source.path}: missing {', '.join(missing)}"
        )

    prompt_type = str(prompt_type)
    if prompt_type not in ALLOWED_PROMPT_TYPES:
        raise ValueError(
            f"Bad prompt_type={prompt_type!r} in {source.path} row #{row_index}; "
            f"expected one of {sorted(ALLOWED_PROMPT_TYPES)}"
        )

    blind: dict[str, Any] = {
        "sample_id": "",  # filled after global shuffle
        "prompt_id": str(prompt_id),
        "prompt_type": prompt_type,
        "prompt": str(prompt),
        "generated_text": "" if generated_text is None else str(generated_text),
    }

    if max_new_tokens is not None:
        try:
            blind["max_new_tokens"] = int(max_new_tokens)
        except Exception:
            blind["max_new_tokens"] = max_new_tokens

    original_sample_id = get_first(raw, ("sample_id", "generation_id", "id"), default=None)

    # Store all source-identifying data only inside the private map.
    hidden = {}
    hidden.update(source.meta)
    hidden.update(pick_hidden_meta(raw))

    source_key = sanitize_filename(source.meta.get("source_key") or source.path.stem)
    entry: dict[str, Any] = {
        "source_key": source_key,
        "source_file": str(source.path),
        "source_stem": source.path.stem,
        "row_index": row_index,
        "original_sample_id": None if original_sample_id is None else str(original_sample_id),
        "original_prompt_id": str(prompt_id),
        "hidden": hidden,
        "original": raw,
    }

    return PackedSample(blind=blind, entry=entry)


def make_blind_sample_id(i: int, salt: str) -> str:
    # The visible id is opaque and independent of model name/source filename.
    # The counter makes it readable; the hash makes accidental collisions across files/runs unlikely.
    return f"S{i:06d}_{short_hash(salt + ':' + str(i), 6)}"


def cmd_prepare(args: argparse.Namespace) -> int:
    run_dir = ensure_dir(args.run_dir, die=True).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else (run_dir / "meta.json")
    if args.meta is None and not meta_path.is_file():
        meta_path = None

    sources, meta = discover_source_files(run_dir, meta_path, args.input_files)
    if not sources:
        raise RuntimeError(
            f"No generation JSON/JSONL files found in {run_dir}. "
            f"Pass them explicitly with --input-files."
        )

    packed: list[PackedSample] = []
    per_source_counts: dict[str, int] = {}

    for sf in sources:
        rows = read_json_records(sf.path)
        if not rows:
            print(f"WARN: empty source file: {sf.path}")
            continue

        # A lot of generation files repeat model/hash fields in every row. If file-level meta
        # was not found in meta.json, infer it from the first row for restoration only.
        if not sf.meta:
            sf.meta.update(pick_hidden_meta(rows[0]))
            sf.meta.setdefault("source_key", sf.path.stem)

        for i, row in enumerate(rows):
            ps = normalize_for_judge(row, sf, i)
            packed.append(ps)
            per_source_counts[ps.entry["source_key"]] = per_source_counts.get(ps.entry["source_key"], 0) + 1

    if not packed:
        raise RuntimeError("All source files were empty; nothing to prepare")

    rnd = random.Random(args.seed)
    rnd.shuffle(packed)

    salt = args.salt or short_hash(str(run_dir) + now_iso() + str(args.seed), 12)
    blind_rows: list[dict[str, Any]] = []
    map_entries: dict[str, dict[str, Any]] = {}

    for i, ps in enumerate(packed, start=1):
        sid = make_blind_sample_id(i, salt)
        ps.blind["sample_id"] = sid

        # If requested, remove max_new_tokens even when present. This is optional paranoia.
        if args.drop_max_new_tokens:
            ps.blind.pop("max_new_tokens", None)

        # Judge-visible fields only, in stable order.
        blind = {k: ps.blind[k] for k in JUDGE_FIELDS if k in ps.blind}
        blind_rows.append(blind)

        entry = dict(ps.entry)
        if args.no_store_original:
            entry.pop("original", None)
        map_entries[sid] = entry

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    map_path = Path(args.map).expanduser()
    if not map_path.is_absolute():
        map_path = (Path.cwd() / map_path).resolve()

    n_written = write_jsonl(out_path, blind_rows)

    chunks: list[str] = []
    if args.chunk_size and args.chunk_size > 0:
        stem = out_path.with_suffix("")
        suffix = out_path.suffix or ".jsonl"
        for part_i, start in enumerate(range(0, len(blind_rows), args.chunk_size), start=1):
            part_path = stem.with_name(f"{stem.name}.part{part_i:03d}{suffix}")
            write_jsonl(part_path, blind_rows[start : start + args.chunk_size])
            chunks.append(str(part_path))

    private_map = {
        "version": 1,
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "meta_path": str(meta_path) if meta_path else None,
        "seed": args.seed,
        "salt": salt,
        "total_samples": len(blind_rows),
        "sources": [
            {
                "source_key": sf.key,
                "path": str(sf.path),
                "meta": sf.meta,
                "count": per_source_counts.get(sf.key, 0),
            }
            for sf in sources
        ],
        "chunks": chunks,
        "entries": map_entries,
        "meta": meta if args.store_meta else None,
    }
    save_json(map_path, private_map)

    print(f"Prepared blind JSONL: {out_path} ({n_written} samples)")
    print(f"Private restore map:  {map_path}")
    if chunks:
        print(f"Chunks:               {len(chunks)}")
    print("Source counts:")
    for k in sorted(per_source_counts):
        print(f"  {k}: {per_source_counts[k]}")
    return 0


# =======================================================
# Restoring judged output
# =======================================================

def validate_judged_row(row: dict[str, Any], where: str) -> list[str]:
    warnings: list[str] = []

    sid = row.get("sample_id")
    if not isinstance(sid, str) or not sid:
        warnings.append(f"{where}: missing/invalid sample_id")

    scores = row.get("scores")
    if not isinstance(scores, dict):
        warnings.append(f"{where}: missing/invalid scores object")
    else:
        for name in SCORE_FIELDS:
            v = scores.get(name)
            if not isinstance(v, int) or not (1 <= v <= 5):
                warnings.append(f"{where}: bad score {name}={v!r}")

    flags = row.get("flags")
    if not isinstance(flags, dict):
        warnings.append(f"{where}: missing/invalid flags object")
    else:
        for name in FLAG_FIELDS:
            v = flags.get(name)
            if not isinstance(v, bool):
                warnings.append(f"{where}: bad flag {name}={v!r}")

    brief_reason = row.get("brief_reason")
    if not isinstance(brief_reason, str):
        warnings.append(f"{where}: missing/invalid brief_reason")

    return warnings


def load_judged_files(paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for p_raw in paths:
        p = ensure_file(p_raw, die=True)
        cur = read_json_records(p)
        for i, row in enumerate(cur, start=1):
            warnings.extend(validate_judged_row(row, f"{p}:{i}"))
            rows.append(row)
    return rows, warnings


def merged_restored_row(judged: dict[str, Any], entry: dict[str, Any], include_hidden: bool) -> dict[str, Any]:
    original = entry.get("original") if isinstance(entry.get("original"), dict) else {}
    hidden = entry.get("hidden") if isinstance(entry.get("hidden"), dict) else {}

    out: dict[str, Any] = dict(original)

    # Preserve both ids: original id for local debugging and blind id for traceability.
    out["judge_sample_id"] = judged.get("sample_id")
    if entry.get("original_sample_id") is not None:
        out["original_sample_id"] = entry.get("original_sample_id")

    # Keep the judge-visible fields if original was not stored in the private map.
    for name in ("prompt_id", "prompt_type", "prompt", "generated_text", "max_new_tokens"):
        if name in judged and name not in out:
            out[name] = judged[name]

    out["scores"] = judged.get("scores")
    out["flags"] = judged.get("flags")
    out["brief_reason"] = judged.get("brief_reason")

    if include_hidden:
        # Useful for later aggregation; this data was never sent to the judge.
        for k, v in hidden.items():
            if k not in out:
                out[k] = v
        out["source_file"] = entry.get("source_file")
        out["source_key"] = entry.get("source_key")
        out["source_row_index"] = entry.get("row_index")

    return out



def choose_restore_group_key(entry: dict[str, Any], group_by: str = "auto") -> str:
    source_file = entry.get("source_file")
    source_stem = entry.get("source_stem")
    source_key = entry.get("source_key")

    if group_by == "source_key":
        return sanitize_filename(source_key or source_stem or "unknown")

    if group_by == "source_stem":
        if source_stem:
            return sanitize_filename(source_stem)
        if source_file:
            return sanitize_filename(Path(str(source_file)).stem)
        return sanitize_filename(source_key or "unknown")

    if group_by == "source_file":
        if source_file:
            return sanitize_filename(Path(str(source_file)).stem)
        return sanitize_filename(source_stem or source_key or "unknown")

    # auto: use source_key unless it is a generic metadata field name like
    # output_file; then fall back to the actual source filename stem.
    return effective_source_key(source_key, source_stem=source_stem, source_file=source_file)


def cmd_restore(args: argparse.Namespace) -> int:
    map_path = ensure_file(args.map, die=True).resolve()
    private_map = load_json(map_path)
    if not isinstance(private_map, dict) or not isinstance(private_map.get("entries"), dict):
        raise ValueError(f"Bad private map format: {map_path}")

    entries: dict[str, Any] = private_map["entries"]
    judged_rows, warnings = load_judged_files(args.judged)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    missing_in_map: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    for judged in judged_rows:
        sid = str(judged.get("sample_id") or "")
        if not sid:
            continue
        if sid in seen:
            duplicates.append(sid)
        seen.add(sid)

        entry = entries.get(sid)
        if not isinstance(entry, dict):
            missing_in_map.append(sid)
            continue

        group_key = choose_restore_group_key(entry, args.group_by)
        grouped.setdefault(group_key, []).append(merged_restored_row(judged, entry, args.include_hidden))

    written_files: dict[str, str] = {}
    for group_key, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (str(r.get("prompt_id", "")), int(r.get("source_row_index", 0) or 0)))
        out_path = out_dir / f"{args.prefix}{group_key}.jsonl"
        write_jsonl(out_path, rows)
        written_files[group_key] = str(out_path)

    combined_path = None
    if args.combined:
        combined_rows: list[dict[str, Any]] = []
        for group_key in sorted(grouped):
            combined_rows.extend(grouped[group_key])
        combined_path = out_dir / args.combined
        write_jsonl(combined_path, combined_rows)

    not_judged = sorted(set(entries.keys()) - seen)

    report = {
        "created_at": now_iso(),
        "map_path": str(map_path),
        "judged_files": [str(Path(p)) for p in args.judged],
        "total_judged_rows": len(judged_rows),
        "total_map_entries": len(entries),
        "restored_rows": sum(len(v) for v in grouped.values()),
        "written_files": written_files,
        "combined_file": str(combined_path) if combined_path else None,
        "warnings": warnings,
        "duplicate_sample_ids": duplicates,
        "missing_in_map": missing_in_map,
        "not_judged_count": len(not_judged),
        "not_judged_first_50": not_judged[:50],
    }
    report_path = out_dir / "restore_report.json"
    save_json(report_path, report)

    print(f"Restored rows: {report['restored_rows']}/{len(entries)}")
    print(f"Output dir:     {out_dir}")
    for group_key, path in written_files.items():
        print(f"  {group_key}: {path}")
    if combined_path:
        print(f"Combined:       {combined_path}")
    print(f"Report:         {report_path}")
    if warnings:
        print(f"WARNINGS:       {len(warnings)} (see restore_report.json)")
    if missing_in_map:
        print(f"Missing in map: {len(missing_in_map)}")
    if not_judged:
        print(f"Not judged:     {len(not_judged)}")
    return 0


# =======================================================
# CLI
# =======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Create blind shuffled JSONL for LLM judging and restore judged results back to model files."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", aliases=["pack", "anonymize"], help="make anonymous shuffled JSONL")
    p.add_argument("run_dir", type=str, help="directory like runs/E_1 with meta.json and generation files")
    p.add_argument("--meta", type=str, default="", help="path to meta.json; default: <run_dir>/meta.json if it exists")
    p.add_argument("--input-files", type=str, nargs="*", default=None, help="explicit generation files; bypass/fix meta discovery")
    p.add_argument("--out", type=str, default="data/blind_eval.jsonl", help="blind JSONL output path; relative to current working directory by default")
    p.add_argument("--map", type=str, default="data/blind_eval_map.json", help="private map path; relative to current working directory by default")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed")
    p.add_argument("--salt", type=str, default="", help="optional salt for anonymous sample_id generation")
    p.add_argument("--chunk-size", type=int, default=0, help="also split blind file into chunks of this many rows")
    p.add_argument("--drop-max-new-tokens", action="store_true", help="do not send max_new_tokens to judge")
    p.add_argument("--no-store-original", action="store_true", help="do not store original rows in the private map")
    p.add_argument("--store-meta", action="store_true", help="store full meta.json inside private map")
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("restore", aliases=["unpack", "deanonymize"], help="restore judged JSONL to per-model files")
    r.add_argument("--map", type=str, required=True, help="private map produced by prepare")
    r.add_argument("--judged", type=str, nargs="+", required=True, help="judge JSONL output file(s); parts are allowed")
    r.add_argument("--out-dir", type=str, default="res/restored_judged", help="output directory; relative to current working directory by default")
    r.add_argument("--prefix", type=str, default="judged_", help="prefix for restored per-source files")
    r.add_argument("--combined", type=str, default="judged_all.jsonl", help="combined output filename; empty string disables it")
    r.add_argument("--include-hidden", action="store_true", help="include model/hash/source fields in restored rows")
    r.add_argument(
        "--group-by",
        choices=["auto", "source_key", "source_stem", "source_file"],
        default="auto",
        help=(
            "how to split restored outputs; auto uses source_key unless it is a "
            "generic meta field name like output_file, then uses source filename stem"
        ),
    )
    r.set_defaults(func=cmd_restore)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.cmd in ("prepare", "pack", "anonymize"):
        args.meta = args.meta or None
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
