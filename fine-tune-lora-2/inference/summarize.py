from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


JUDGED_SUFFIX = "_judged.jsonl"
TYPE_ORDER = ["continuation", "instruct", "constraints", "ood_simple"]


# =======================================================


def ensure_file(path: Path | str, strict: bool = True) -> Path:
    path = Path(path)

    if not path.exists() or not path.is_file():
        if strict:
            raise FileNotFoundError(f"File on path: {path} not found")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    return path


def ensure_dir(path: Path | str, strict: bool = False) -> Path:
    path = Path(path)

    if not path.exists() or not path.is_dir():
        if strict:
            raise FileNotFoundError(f"Dir on path: {path} not found")
        path.mkdir(parents=True, exist_ok=True)

    return path


def resolve_root(root: str) -> Path:
    """
    root is the project root, i.e. the directory that contains inference/.

    Empty root is resolved so that both variants work:
      python inference/summarize.py
      cd inference && python summarize.py
    """
    if root:
        return Path(root).expanduser().resolve()

    cwd = Path.cwd().resolve()

    if (cwd / "inference" / "res").is_dir():
        return cwd

    if cwd.name == "inference" and (cwd / "res").is_dir():
        return cwd.parent

    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "inference" and (script_dir / "res").is_dir():
        return script_dir.parent

    return cwd


def read_json(path: Path) -> dict[str, Any]:
    ensure_file(path, strict=True)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def read_jsonl(path: Path, strict_json: bool) -> list[dict[str, Any]]:
    ensure_file(path, strict=True)

    res: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, row in enumerate(f, start=1):
            row = row.strip()
            if not row:
                continue

            try:
                item = json.loads(row)
            except Exception:
                if strict_json:
                    raise
                print(f"[WARN] skip bad json line: {path}:{line_no}")
                continue

            if not isinstance(item, dict):
                if strict_json:
                    raise ValueError(f"Expected object in {path}:{line_no}")
                print(f"[WARN] skip non-object line: {path}:{line_no}")
                continue

            res.append(item)

    return res


# =======================================================


def get_exp_dirs(root: Path, experiments: str) -> list[Path]:
    res_root = root / "inference" / "res"
    ensure_dir(res_root, strict=True)

    if experiments and experiments != "all":
        names = [x.strip() for x in experiments.split(",") if x.strip()]
        exp_dirs = [res_root / name for name in names]
        for p in exp_dirs:
            ensure_dir(p, strict=True)
        return exp_dirs

    exp_dirs = sorted([p for p in res_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    # fallback: allow inference/res/*.jsonl directly, without experiment subdirs
    if not exp_dirs and find_judged_files(res_root):
        exp_dirs = [res_root]

    return exp_dirs


def find_judged_files(exp_res_path: Path, judged_suffix: str = JUDGED_SUFFIX) -> list[Path]:
    files = sorted(exp_res_path.glob(f"*{judged_suffix}"), key=lambda p: p.name)

    # fallback for names like "model.judged.jsonl" or "model-judged.jsonl"
    if not files:
        files = sorted(
            [p for p in exp_res_path.glob("*.jsonl") if "judged" in p.stem],
            key=lambda p: p.name,
        )

    return files


def model_name_from_judged_file(path: Path, judged_suffix: str = JUDGED_SUFFIX) -> str:
    if path.name.endswith(judged_suffix):
        return path.name[: -len(judged_suffix)]

    stem = path.stem
    return stem.split("judged", 1)[0].rstrip("_-")


def load_exp_meta(root: Path, exp_name: str) -> dict[str, Any]:
    meta_path = root / "inference" / "runs" / exp_name / "meta.json"
    if not meta_path.is_file():
        return {}
    return read_json(meta_path)


def get_model_meta(exp_meta: dict[str, Any], model: str) -> dict[str, Any]:
    direct = exp_meta.get(model)
    if isinstance(direct, dict):
        return direct

    # fallback by output_file stem
    for _name, data in exp_meta.items():
        if not isinstance(data, dict):
            continue
        output_file = data.get("output_file", "")
        if Path(str(output_file)).stem == model:
            return data

    return {}


def get_models(root: Path, experiments: str, judged_suffix: str) -> list[dict[str, Any]]:
    res: list[dict[str, Any]] = []

    for exp_res_path in get_exp_dirs(root, experiments):
        exp_name = exp_res_path.name
        exp_meta = load_exp_meta(root, exp_name)

        for f in find_judged_files(exp_res_path, judged_suffix=judged_suffix):
            model = model_name_from_judged_file(f, judged_suffix=judged_suffix)
            meta = get_model_meta(exp_meta, model)

            res.append(
                {
                    "exp_name": exp_name,
                    "exp_path": exp_res_path,
                    "model": model,
                    "judged_file": f,
                    "model_hash": str(meta.get("model_hash") or ""),
                    "model_path": str(meta.get("model_path") or ""),
                }
            )

    return res


# =======================================================


def add_ordered(xs: list[str], x: str) -> None:
    if x and x not in xs:
        xs.append(x)


def prompt_type_sort_key(x: str) -> tuple[int, str]:
    if x in TYPE_ORDER:
        return (TYPE_ORDER.index(x), x)
    return (len(TYPE_ORDER), x)


def safe_col_part(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\W+", "_", x)
    x = x.strip("_")
    return x or "unknown"


def to_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None

    try:
        v = float(x)
    except Exception:
        return None

    if math.isnan(v):
        return None

    return v


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        return bool(x)

    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y", "да"}

    return bool(x)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def std(xs: list[float]) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return statistics.stdev(xs)


def fmt(v: Any) -> str:
    if v is None:
        return ""

    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6f}".rstrip("0").rstrip(".")

    return str(v)


def get_scores(sample: dict[str, Any]) -> dict[str, Any]:
    scores = sample.get("scores", {})
    return scores if isinstance(scores, dict) else {}


def get_bool_tags(sample: dict[str, Any]) -> dict[str, Any]:
    """
    In current judged files this field is named "flags".
    "tags" is also supported because it is easy to rename it upstream later.
    If both exist, both are merged. On key conflict flags wins.
    """
    res: dict[str, Any] = {}

    tags = sample.get("tags", {})
    if isinstance(tags, dict):
        res.update(tags)

    flags = sample.get("flags", {})
    if isinstance(flags, dict):
        res.update(flags)

    return res


def collect_schema(
    records: list[dict[str, Any]],
    score_names: list[str],
    bool_names: list[str],
    prompt_types: list[str],
) -> None:
    for sample in records:
        add_ordered(prompt_types, str(sample.get("prompt_type") or "unknown"))

        for k in get_scores(sample).keys():
            add_ordered(score_names, str(k))

        for k in get_bool_tags(sample).keys():
            add_ordered(bool_names, str(k))


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["all"] = list(records)

    for sample in records:
        prompt_type = str(sample.get("prompt_type") or "unknown")
        groups[prompt_type].append(sample)

    return dict(groups)


def aggregate_group(
    records: list[dict[str, Any]],
    score_names: list[str],
    bool_names: list[str],
    prefix: str | None = None,
) -> dict[str, Any]:
    res: dict[str, Any] = {}

    def col(name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    prompt_ids = {
        str(x.get("prompt_id"))
        for x in records
        if x.get("prompt_id") is not None
    }

    res[col("n")] = len(records)
    res[col("prompts")] = len(prompt_ids)

    for score_name in score_names:
        vals: list[float] = []
        for sample in records:
            scores = get_scores(sample)
            v = to_float(scores.get(score_name))
            if v is not None:
                vals.append(v)

        res[col(f"{score_name}_mean")] = mean(vals)
        res[col(f"{score_name}_std")] = std(vals)

    for bool_name in bool_names:
        vals: list[bool] = []
        for sample in records:
            flags = get_bool_tags(sample)
            if bool_name in flags:
                vals.append(to_bool(flags[bool_name]))

        # For each flag keep only one statistic: how many times it was True.
        # No *_false / *_true_rate columns, so the table stays compact.
        res[col(f"flag_{bool_name}")] = sum(vals)

    return res


def summarize_model(
    info: dict[str, Any],
    records: list[dict[str, Any]],
    score_names: list[str],
    bool_names: list[str],
    prompt_types: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {
        "exp_name": info["exp_name"],
        "model": info["model"],
        "model_hash": info.get("model_hash", ""),
        "model_path": info.get("model_path", ""),
        "judged_file": str(info["judged_file"]),
    }

    groups = group_records(records)
    ordered_types = ["all"] + sorted(prompt_types, key=prompt_type_sort_key)

    wide = dict(base)
    wide["n_total"] = len(records)
    wide["prompt_types"] = ",".join([x for x in ordered_types if x != "all"])

    long_rows: list[dict[str, Any]] = []

    for prompt_type in ordered_types:
        cur_records = groups.get(prompt_type, [])
        prefix = safe_col_part(prompt_type)

        wide.update(
            aggregate_group(
                records=cur_records,
                score_names=score_names,
                bool_names=bool_names,
                prefix=prefix,
            )
        )

        long = dict(base)
        long["prompt_type"] = prompt_type
        long.update(
            aggregate_group(
                records=cur_records,
                score_names=score_names,
                bool_names=bool_names,
                prefix=None,
            )
        )
        long_rows.append(long)

    return wide, long_rows


# =======================================================


def wide_headers(
    score_names: list[str],
    bool_names: list[str],
    prompt_types: list[str],
) -> list[str]:
    headers = [
        "exp_name",
        "model",
        "model_hash",
        "model_path",
        "judged_file",
        "n_total",
        "prompt_types",
    ]

    ordered_types = ["all"] + sorted(prompt_types, key=prompt_type_sort_key)

    for prompt_type in ordered_types:
        prefix = safe_col_part(prompt_type)

        headers += [
            f"{prefix}_n",
            f"{prefix}_prompts",
        ]

        for score_name in score_names:
            headers += [
                f"{prefix}_{score_name}_mean",
                f"{prefix}_{score_name}_std",
            ]

        for bool_name in bool_names:
            headers.append(f"{prefix}_flag_{bool_name}")

    return headers


def long_headers(score_names: list[str], bool_names: list[str]) -> list[str]:
    headers = [
        "exp_name",
        "model",
        "model_hash",
        "model_path",
        "judged_file",
        "prompt_type",
        "n",
        "prompts",
    ]

    for score_name in score_names:
        headers += [
            f"{score_name}_mean",
            f"{score_name}_std",
        ]

    for bool_name in bool_names:
        headers.append(f"flag_{bool_name}")

    return headers


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # keep any unexpected extra columns instead of silently dropping them
    seen = set(headers)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()

        for row in rows:
            w.writerow({h: fmt(row.get(h)) for h in headers})


# =======================================================


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        "--runs-root",
        dest="root",
        type=str,
        default="",
        help="Project root, i.e. directory that contains inference/. Empty = autodetect.",
    )
    ap.add_argument(
        "--experiments",
        type=str,
        default="",
        help='Experiment names separated by comma, e.g. "E_1,E_2". Empty or "all" = all.',
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path. Empty = inference/res/summary.csv for wide, summary_long.csv for long.",
    )
    ap.add_argument(
        "--format",
        choices=["wide", "long"],
        default="long",
        help="wide = one row per model; long = one row per model and prompt_type. Default = long.",
    )
    ap.add_argument(
        "--judged-suffix",
        type=str,
        default=JUDGED_SUFFIX,
        help=f"Judged files suffix, default {JUDGED_SUFFIX!r}.",
    )
    ap.add_argument(
        "--strict-json",
        action="store_true",
        help="Fail on malformed JSONL lines instead of skipping them.",
    )

    args = ap.parse_args()

    root = resolve_root(args.root)

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        name = "summary.csv" if args.format == "wide" else "summary_long.csv"
        out_path = root / "inference" / "res" / name

    model_files = get_models(
        root=root,
        experiments=args.experiments,
        judged_suffix=args.judged_suffix,
    )

    if not model_files:
        raise FileNotFoundError(
            f"No judged files found under {root / 'inference' / 'res'}"
        )

    score_names: list[str] = []
    bool_names: list[str] = []
    prompt_types: list[str] = []

    loaded: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for info in model_files:
        records = read_jsonl(info["judged_file"], strict_json=args.strict_json)
        collect_schema(
            records=records,
            score_names=score_names,
            bool_names=bool_names,
            prompt_types=prompt_types,
        )
        loaded.append((info, records))

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for info, records in loaded:
        wide, long = summarize_model(
            info=info,
            records=records,
            score_names=score_names,
            bool_names=bool_names,
            prompt_types=prompt_types,
        )
        wide_rows.append(wide)
        long_rows.extend(long)

    if args.format == "wide":
        rows = sorted(wide_rows, key=lambda x: (x["exp_name"], x["model"]))
        headers = wide_headers(score_names, bool_names, prompt_types)
    else:
        rows = sorted(
            long_rows,
            key=lambda x: (
                prompt_type_sort_key(str(x["prompt_type"])),
                x["model"],
                x["exp_name"],
            ),
        )
        headers = long_headers(score_names, bool_names)

    write_csv(out_path, rows, headers)

    print(f"root:       {root}")
    print(f"models:     {len(model_files)}")
    print(f"format:     {args.format}")
    print(f"wrote:      {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
