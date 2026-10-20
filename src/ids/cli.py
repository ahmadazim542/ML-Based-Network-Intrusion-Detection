"""Command line entry point:  ids <step> [--config configs/x.yaml] [options]"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from . import experiments as ex
from .config import load_config

STEPS = ["prepare", "benchmark", "stats", "deploy", "explain", "cross", "loao", "report", "all"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ids", description=__doc__)
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--config", default=None, help="YAML layered on top of configs/default.yaml")
    ap.add_argument("--models", nargs="+", help="subset of models for `benchmark`")
    ap.add_argument("--strategies", nargs="+", help="subset of imbalance strategies for `benchmark`")
    ap.add_argument("--task", choices=["multiclass", "binary"])
    ap.add_argument("--split", choices=["random", "time_block"])
    ap.add_argument("--skip-cv", action="store_true", help="skip per-fold CV (faster, no t-test)")
    ap.add_argument("--force", action="store_true", help="rebuild the prepared dataset cache")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    overrides = {}
    if a.task:
        overrides["task"] = a.task
    if a.split:
        overrides["split"] = {"strategy": a.split}
    cfg = load_config(a.config, **overrides)

    steps = STEPS[:-1] if a.step == "all" else [a.step]
    log = logging.getLogger("ids")
    failed = []
    for step in steps:
        log.info("---- %s", step)
        t0 = time.perf_counter()
        try:
            _run_step(step, cfg, a)
        except Exception:
            if a.step != "all":
                raise
            # In `all`, one failing step (e.g. cross-dataset data not downloaded yet) must not
            # throw away hours of work: log it and carry on with the independent steps.
            log.exception("step %s failed; continuing", step)
            failed.append(step)
            if step in ("prepare", "benchmark"):
                break  # everything after depends on these
        log.info("---- %s done in %.1f min", step, (time.perf_counter() - t0) / 60)
    if failed:
        log.error("failed steps: %s", ", ".join(failed))
        sys.exit(1)


def _run_step(step: str, cfg, a) -> None:
    if step == "prepare":
        ex.prepare(cfg, force=a.force)
    elif step == "benchmark":
        print(ex.benchmark(cfg, a.models, a.strategies, a.skip_cv)
              .reindex(columns=["model", "strategy", "f1_macro", "bin_recall", "bin_fpr", "train_seconds"])
              .to_string(index=False))
    elif step == "stats":
        print(ex.run_stats(cfg).to_string(index=False))
    elif step == "deploy":
        print(ex.run_deploy(cfg).to_string(index=False))
    elif step == "explain":
        print(ex.run_explain(cfg))
    elif step == "cross":
        print(ex.run_cross_dataset(cfg).to_string(index=False))
    elif step == "loao":
        print(ex.run_loao(cfg).to_string(index=False))
    elif step == "report":
        print(ex.report(cfg).to_string(index=False))

if __name__ == "__main__":
    main()
