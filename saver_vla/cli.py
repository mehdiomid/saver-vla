"""Command line: `saver run`, `saver sweep`, `saver report`.

    saver run configs/phase651.yaml --set recovery.model=gpt-5.2
    saver sweep configs/sweep_reliability.yaml --dry-run
    saver report runs/reliability_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="saver", description="SAVER-VLA agentic failure recovery")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one episode")
    run.add_argument("config")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    run.add_argument("--out", help="output folder (default: <output_dir>/<name>/<timestamp>)")
    run.add_argument("--quiet", action="store_true", help="do not print events")

    sweep = sub.add_parser("sweep", help="run a reliability sweep")
    sweep.add_argument("config")
    sweep.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    sweep.add_argument("--dry-run", action="store_true", help="list the episodes without running them")

    report = sub.add_parser("report", help="summarize all results under a folder")
    report.add_argument("run_dir")

    args = parser.parse_args(argv)

    if args.command == "run":
        from .config import load_config
        from .runner import make_backend, run_episode

        cfg = load_config(args.config, args.set)
        out = Path(args.out) if args.out else Path(cfg.output_dir) / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
        backend = make_backend(cfg)
        try:
            result = run_episode(cfg, backend, out_dir=out, echo=not args.quiet)
        finally:
            backend.close()
        print(json.dumps({k: result[k] for k in ("success", "outcome", "recovery_outcome", "decisions", "steps")}))
        print(f"Saved to {out}")
        return 0

    if args.command == "sweep":
        from .sweep import run_sweep

        run_sweep(args.config, args.set, dry_run=args.dry_run)
        return 0

    if args.command == "report":
        from .metrics import write_report

        write_report(args.run_dir)
        print((Path(args.run_dir) / "report.md").read_text())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
