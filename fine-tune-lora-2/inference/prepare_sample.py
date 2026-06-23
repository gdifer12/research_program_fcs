from __future__ import annotations
import argparse
import os
import subprocess
import csv
import json
import hashlib
from pathlib import Path


def ensure_file(path: Path | str, strict: bool = True):
    path = Path(path) 
    
    if not path.exists() or not path.is_file():
        if strict:
            raise FileNotFoundError(f"File on path: {path} not found")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

def ensure_dir(path: Path | str, strict: bool = False):
    path = Path(path) 
    
    if not path.exists() or not path.is_dir():
        if strict:
            raise FileNotFoundError(f"Dir on path: {path} not found")
        path.mkdir(parents=True, exist_ok=True)
        

def small_hash(s: str, n: int = 10) -> str:
    return hashlib.blake2s(s.encode("utf-8"), digest_size=16).hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--eval-script', type=str, help='Path to bash-like script that will inference model')
    ap.add_argument('--summary', type=str, help='CSV file created by summarize.py with model pathes by model names')
    ap.add_argument('--models', type=str, help='List of models to eval devided by comma')
    ap.add_argument('--prompts', type=str, help='Path of prompts file')
    ap.add_argument('--config', type=str, help='Config file to all inference', default="")
    ap.add_argument('--out-dir', type=str, help='Path to dir for result files')
    
    ap.add_argument('--project-dir', type=str, default=None)
    ap.add_argument('--repo-dir', type=str, default=None)
    ap.add_argument('--module-set', type=str, default=None)
    ap.add_argument('--envname', type=str, default=None)
    
    args = ap.parse_args()

    models = list([model.strip() for model in args.models.split(',')])
    
    eval_script = Path(args.eval_script)
    summary_path = Path(args.summary)
    prompts_path = Path(args.prompts)
    config_path = Path(args.config) if args.config else None
    out_dir = Path(args.out_dir)
    
    ensure_file(eval_script)
    ensure_file(summary_path)
    ensure_file(prompts_path)
    if config_path is not None: ensure_file(config_path)
    ensure_dir(out_dir, strict=False)
    
    summary = dict()
    
    requested_models = [model for model in models if ':' not in model]
    
    project_root = summary_path.parent.parent
    
    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row.get('name')
            if name not in requested_models:
                continue
            
            run_dir = Path(row.get('run_dir', ''))
            if not run_dir.is_absolute():
                run_dir = project_root / run_dir
            run_dir /= 'out'
            
            fname = 'ckpt_best.pt'
            if not (run_dir / fname).exists():
                fname = 'ckpt.pt'
            summary[name] = str(run_dir / fname)
            ensure_file(summary[name])
    
    for i in range(len(models)):
        model = models[i]
        if ':' in model:
            model, path = model.split(':', 1)
            ensure_file(path)
            models[i] = model
            summary[model] = path
        elif model not in summary:
            raise ValueError(f"No such model: {model} in summary: {summary_path}")
        
    hashes = {}
    for model in models:
        hashes[model] = small_hash(f"{model}:{summary[model]}")
        
        export_args = [
            f"MODEL_PATH={summary[model]}",
            f"PROMPTS_PATH={prompts_path}",
            f"OUT_DIR={out_dir}",
            f"OUT_FILENAME={model}.jsonl",
            f"JSON_HEADER={json.dumps({'model_hash': hashes[model]}, separators=(',', ':'))}",
        ]
        if config_path is not None:
            export_args.append(f"CONFIG_PATH={config_path}")
        
        if args.project_dir is not None:
            export_args.append(f"PROJECT_DIR={args.project_dir}")
        if args.repo_dir is not None:
            export_args.append(f"REPO_DIR={args.repo_dir}")
        if args.module_set is not None:
            export_args.append(f"MODULE_SET={args.module_set}")
        if args.envname is not None:
            export_args.append(f"ENVNAME={args.envname}")
    
        cmd = [
            'sbatch',
            f'--job-name=nanogpt-{model}',
            f'--export=ALL,{",".join(export_args)}',
            f'{eval_script}'
        ]
        subprocess.run(cmd, check=True)
        
        # env = os.environ.copy()
        # for arg in export_args:
        #     key, value = arg.split("=", 1)
        #     env[key] = value
        # cmd = [
        #     "bash",
        #     str(eval_script),
        # ]
        # subprocess.run(cmd, check=True, env=env)
        
    meta = {
        model: {
            "model_hash": hashes[model],
            "model_path": summary[model],
            "output_file": f"{model}.jsonl",
        }
        for model in models
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write('\n')

if __name__ == "__main__":
    raise SystemExit(main())

