#!/usr/bin/env python3
r"""Run harness ablation experiments across 4 tasks x 6 default configs.

This is a sibling of ``run_ablations.py``. The *only* thing that differs is
the ablation semantics:

  - baseline:     enabled=false, h2=h3=h4=h5=true  (no harness baseline)
  - with_harness: enabled=true,  h2=h3=h4=h5=true  (everything on)
  - no_h2:        h2=false, h3=h4=h5=true          (turn OFF h2 only)
  - no_h3:        h3=false, h2=h4=h5=true          (turn OFF h3 only)
  - no_h4:        h4=false, h2=h3=h5=true          (turn OFF h4 only)
  - no_h5:        h5=false, h2=h3=h4=true          (turn OFF h5 only)
  - only-h3-h5:   h2=h4=false, h3=h5=true          (turn ON h3+h5 only)

Each ablation round now: (1) patches configs/tasks/<task>.yaml for every
selected task (only the <task>-std profile, same state-machine as
run_ablations.py, plus enabled:), (2) patches configs/assignments/<task>.yaml
for every selected task so exactly one output: line is active, (3) docker
compose up -d --force-recreate redis controller plus all selected <task>-std
workers, (4) runs assigners for all selected tasks in parallel. WebShop gets
an extra boot delay before its assigner starts because it registers slowly.

Only the -std profile is exercised; -env_train profiles are left alone.
The patched output: suffix on each row uses this script's labels
(baseline / with_harness / no_h2 / ... / no_h5 / only-h3-h5), so
previously-produced directories from run_ablations.py (with_h2 / with_h3 / ...)
stay untouched on disk.

Usage
-----

    cd my_agent_bench
    python run_ablations_leave_one_out.py                # all 4 x 6 = 24 default runs
    python run_ablations_leave_one_out.py --tasks alfworld
    python run_ablations_leave_one_out.py --ablations baseline no_h4
    python run_ablations_leave_one_out.py --dry-run
    python run_ablations_leave_one_out.py --only-edit
    python run_ablations_leave_one_out.py --skip-docker
    python run_ablations_leave_one_out.py --continue-on-error
    python run_ablations_leave_one_out.py --ready-timeout 0

    # 查看 yaml 修改结果
    python run_ablations_leave_one_out_parallel.py --preview-dir ./ablation_tmp/yaml_preview
    # 执行 baseline 和 harness
    python run_ablations_leave_one_out_parallel.py --tasks alfworld dbbench os webshop --ablations baseline with_harness
    # 新增: ALFWorld, 只开启 h3+h5 两层 harness
    python run_ablations_leave_one_out_parallel.py --tasks alfworld --ablations only-h3-h5

    # A6000
    # prompt-evolving
    # 仅查看 yaml 改动是否正确，不实际执行
    python run_ablations_leave_one_out_parallel.py --tasks dbbench os webshop alfworld --ablations only-h3-h5 --preview-dir ./ablation_tmp/yaml_preview_only_h3_h5
    python run_ablations_leave_one_out_parallel.py --tasks alfworld --ablations only-h3-h5 --preview-dir ./ablation_tmp/yaml_preview_only_h3_h5
    # 实际执行
    python run_ablations_leave_one_out_parallel.py --tasks alfworld dbbench os webshop --ablations only-h3-h5
    # 备注: only-h3-h5 会设置为 h2=false, h3=true, h4=false, h5=true


Readiness check
---------------
After 'docker compose up', the <task>-std worker still needs to load its
dataset before registering with the controller. If assigner is started
too early the controller answers ``task <task>-std does not exist`` with
HTTP 400. To avoid that race we poll the controller's GET /get_indices
endpoint and only proceed once it returns 200. Default timeout 300s,
override with --ready-timeout (0 disables).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# All reusable machinery (YAML profile editor, shell runner, controller
# readiness poller, paths/constants) lives in run_ablations.py. We import it
# here so we never fork that logic -- this script only customizes the
# experiment labels, the enabled/h-flag mapping, and the output-label rewrite.
from run_ablations import (
    ASSIGN_DIR,
    COMPOSE_FILE,
    CONTROLLER_URL,
    LOG_DIR,
    ROOT,
    TASKS,
    TASKS_DIR,
    run_cmd,
    wait_for_task_registered,
)


ABLATIONS: List[str] = [
    "baseline",
    "with_harness",
    "no_h2",
    "no_h3",
    "no_h4",
    "no_h5",
    "only-h3-h5",
]
DEFAULT_ABLATIONS: List[str] = [
    "baseline",
    "with_harness",
    "no_h2",
    "no_h3",
    "no_h4",
    "no_h5",
]


def flags_for(ablation: str) -> Dict[str, bool]:
    """Translate an ablation label into the h2/h3/h4/h5 boolean dict.

    Leave-one-out semantics:
      - 'baseline'     -> all True
      - 'with_harness' -> all True
      - 'no_h<N>'      -> only h<N> False, the other three True
      - 'only-h3-h5'   -> only h3 and h5 True
    """
    keys = ("h2", "h3", "h4", "h5")
    if ablation in {"baseline", "with_harness"}:
        return {k: True for k in keys}
    if ablation == "only-h3-h5":
        return {"h2": False, "h3": True, "h4": False, "h5": True}
    if not ablation.startswith("no_h"):
        raise ValueError(f"Unknown ablation label: {ablation!r}")
    off = ablation.split("_", 1)[1]  # e.g. "h2"
    if off not in keys:
        raise ValueError(f"Unknown harness name in ablation: {off!r}")
    return {k: (k != off) for k in keys}


def enabled_for(ablation: str) -> bool:
    """Translate an ablation label into the <task>-std enabled: value."""
    if ablation == "baseline":
        return False
    if ablation in ABLATIONS:
        return True
    raise ValueError(f"Unknown ablation label: {ablation!r}")


def edit_task_yaml(
    path: Path, task: str, flags: Dict[str, bool], enabled: bool
) -> None:
    """Set enabled/h2/h3/h4/h5 inside the <task>-std: profile only."""
    target_header = f"{task}-std:"
    text = path.read_text()
    lines = text.split("\n")

    value_re = re.compile(
        r"^(\s*)(enabled|h[2345]):\s*(true|false)(\s*(?:#.*)?)$",
        re.IGNORECASE,
    )
    top_level_re = re.compile(r"^\S[^:\s]*.*:\s*$")
    replacements = {"enabled": enabled, **flags}

    in_profile = False
    new_lines: List[str] = []
    for line in lines:
        if not in_profile:
            if line.strip() == target_header and not line.startswith((" ", "\t")):
                in_profile = True
            new_lines.append(line)
            continue

        if top_level_re.match(line) and not line.startswith((" ", "\t")):
            in_profile = False
            new_lines.append(line)
            continue

        m = value_re.match(line)
        if m:
            indent, key, _, suffix = m.groups()
            key = key.lower()
            if key in replacements:
                value = "true" if replacements[key] else "false"
                new_lines.append(f"{indent}{key}: {value}{suffix}")
                continue
        new_lines.append(line)

    path.write_text("\n".join(new_lines))


_OUTPUT_RE = re.compile(
    r'^(?P<comment>#\s*)?output:\s*["\'](?P<prefix>.+?)\{TIMESTAMP\}-(?P<suffix>[^"\']+)["\']\s*$'
)


def edit_assignment_yaml(path: Path, target_ablation: str) -> None:
    """Rewrite the block of output: lines so only `target_ablation` is active.

    Locates the contiguous run of ``[#] output: "<prefix>{TIMESTAMP}-<suffix>"``
    lines, keeps the shared <prefix>, and regenerates exactly one line per
    entry in this script's ABLATIONS list. The chosen ablation is uncommented;
    the others are commented out with "# ".

    NOTE: this is a local copy (not imported from run_ablations) because the
    imported version there is hard-wired to run_ablations.ABLATIONS and we
    need it to write *this* script's labels instead.
    """
    text = path.read_text()
    lines = text.split("\n")

    indices: List[int] = []
    prefixes: List[str] = []
    for i, line in enumerate(lines):
        m = _OUTPUT_RE.match(line)
        if m:
            indices.append(i)
            prefixes.append(m.group("prefix"))

    if not indices:
        raise RuntimeError(
            f"Could not locate an 'output: \"...{{TIMESTAMP}}-...\"' block in {path}"
        )

    unique_prefixes = sorted(set(prefixes))
    if len(unique_prefixes) != 1:
        raise RuntimeError(
            f"Output prefixes are inconsistent in {path}: {unique_prefixes}"
        )
    prefix = unique_prefixes[0]

    new_block: List[str] = []
    for a in ABLATIONS:
        head = "" if a == target_ablation else "# "
        new_block.append(f'{head}output: "{prefix}{{TIMESTAMP}}-{a}"')

    start, end = indices[0], indices[-1]
    merged = lines[:start] + new_block + lines[end + 1 :]
    path.write_text("\n".join(merged))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run harness ablation experiments (4 tasks x 6 default configs: "
            "enabled=false baseline, all-on, and disable one hN at a time. "
            "The optional h3+h5-only config is available as only-h3-h5."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="Which tasks to run (default: all).",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=DEFAULT_ABLATIONS,
        choices=ABLATIONS,
        help="Which ablations to run for each task (default: standard 6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen; do not edit files or run any command.",
    )
    parser.add_argument(
        "--only-edit",
        action="store_true",
        help="Only patch the YAML files; do not run docker or the assigner.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip the 'docker compose up' step (assume services are up).",
    )
    parser.add_argument(
        "--webshop-wait",
        type=int,
        default=180,
        help=(
            "Seconds to sleep before starting the webshop assigner in each "
            "parallel ablation round (webshop boots slowly)."
        ),
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=300,
        help=(
            "Max seconds to wait for the <task>-std worker to register with "
            "the controller after 'docker compose up'. 0 disables the wait."
        ),
    )
    parser.add_argument(
        "--ready-interval",
        type=float,
        default=1.0,
        help="Seconds between controller readiness probes.",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default=CONTROLLER_URL,
        help="Controller HTTP base URL used for readiness probes.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next experiment even if the current one fails.",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help=(
            "If set, write patched YAMLs into this directory instead of "
            "editing configs/tasks/*.yaml and configs/assignments/*.yaml. "
            "Implies no docker / assigner runs. Files are written to "
            "<preview_dir>/<task>/<ablation>/{task,assignment}.yaml so you "
            "can inspect all generated variants side-by-side. Originals stay "
            "untouched."
        ),
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_log = LOG_DIR / f"ablation-loo-{session_id}.log"

    def log(msg: str) -> None:
        with log_lock:
            print(msg, flush=True)
            with session_log.open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

    log_lock = threading.Lock()

    preview_root: Path | None = (
        Path(args.preview_dir).resolve() if args.preview_dir else None
    )

    log(f"# Leave-one-out ablation session {session_id}")
    log(f"# tasks       = {args.tasks}")
    log(f"# ablations   = {args.ablations}")
    log(f"# dry_run     = {args.dry_run}")
    log(f"# only_edit   = {args.only_edit}")
    log(f"# skip_docker = {args.skip_docker}")
    log(f"# preview_dir = {preview_root}")

    results: List[Tuple[str, str, int]] = []
    stop_everything = False

    config_paths: Dict[str, Tuple[Path, Path]] = {}
    for task in args.tasks:
        src_task_yaml = TASKS_DIR / f"{task}.yaml"
        src_assign_yaml = ASSIGN_DIR / f"{task}.yaml"
        if not src_task_yaml.exists() or not src_assign_yaml.exists():
            log(
                f"[skip] config missing for task={task}: "
                f"{src_task_yaml} / {src_assign_yaml}"
            )
            continue
        config_paths[task] = (src_task_yaml, src_assign_yaml)

    if not config_paths:
        log("[error] no runnable tasks after checking configs")
        return 1

    def run_task_assigner(task: str, abl: str) -> int:
        """Wait for one task worker and run its assigner."""
        if (
            task == "webshop"
            and args.webshop_wait > 0
            and not args.dry_run
        ):
            log(
                f"[wait] webshop needs extra boot time before assigner; "
                f"sleeping {args.webshop_wait}s..."
            )
            time.sleep(args.webshop_wait)

        if (
            not args.dry_run
            and not args.only_edit
            and args.ready_timeout > 0
        ):
            task_std_name = f"{task}-std"
            ok = wait_for_task_registered(
                task_name=task_std_name,
                controller_url=args.controller_url,
                timeout=float(args.ready_timeout),
                interval=float(args.ready_interval),
                log=log,
            )
            if not ok:
                log(
                    f"[error] controller never saw '{task_std_name}' "
                    f"within {args.ready_timeout}s; skipping assigner."
                )
                return 124

        rc = run_cmd(
            [
                sys.executable,
                "-m",
                "src.assigner",
                "--config",
                f"configs/assignments/{task}.yaml",
            ],
            cwd=ROOT,
            dry_run=args.dry_run,
            log_file=session_log,
        )
        log(f"[done] task={task} ablation={abl} rc={rc}")
        return rc

    for abl in args.ablations:
        if stop_everything:
            break
        flags = flags_for(abl)
        enabled = enabled_for(abl)
        round_tasks = list(config_paths)
        header = (
            f"\n========== ablation={abl}  tasks={round_tasks}  "
            f"enabled={enabled}  flags={{h2:{flags['h2']}, h3:{flags['h3']}, "
            f"h4:{flags['h4']}, h5:{flags['h5']}}}  =========="
        )
        log(header)

        for task, (src_task_yaml, src_assign_yaml) in config_paths.items():
            # Decide where the patched YAMLs go.
            # - preview_dir set -> write to <preview_dir>/<task>/<ablation>/,
            #                      never touch the originals.
            # - otherwise       -> patch the originals in-place.
            if preview_root is not None:
                variant_dir = preview_root / task / abl
                variant_dir.mkdir(parents=True, exist_ok=True)
                task_yaml = variant_dir / "task.yaml"
                assign_yaml = variant_dir / "assignment.yaml"
                if not args.dry_run:
                    task_yaml.write_text(src_task_yaml.read_text())
                    assign_yaml.write_text(src_assign_yaml.read_text())
            else:
                task_yaml = src_task_yaml
                assign_yaml = src_assign_yaml

            if not args.dry_run:
                edit_task_yaml(task_yaml, task, flags, enabled)
                edit_assignment_yaml(assign_yaml, abl)
            log(f"[patched] {task_yaml}")
            log(f"[patched] {assign_yaml}")

        if preview_root is not None or args.only_edit:
            for task in round_tasks:
                results.append((task, abl, 0))
            continue

        if not args.skip_docker:
            compose_services = [
                "redis",
                "controller",
                *[f"{task}-std" for task in round_tasks],
            ]
            rc = run_cmd(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(COMPOSE_FILE),
                    "up",
                    "-d",
                    "--force-recreate",
                    *compose_services,
                ],
                cwd=ROOT,
                dry_run=args.dry_run,
                log_file=session_log,
            )
            if rc != 0:
                log(
                    f"[error] docker compose up failed (rc={rc}) for "
                    f"ablation={abl}; skipping this round."
                )
                for task in round_tasks:
                    results.append((task, abl, rc))
                if not args.continue_on_error:
                    stop_everything = True
                continue

        log(f"[assign] launching {len(round_tasks)} assigner(s) in parallel")
        round_rcs: Dict[str, int] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(round_tasks)
        ) as executor:
            future_to_task = {
                executor.submit(run_task_assigner, task, abl): task
                for task in round_tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    rc = future.result()
                except Exception as exc:
                    log(
                        f"[error] task={task} ablation={abl} raised "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                    rc = 1
                round_rcs[task] = rc
                results.append((task, abl, rc))

        failed = {task: rc for task, rc in round_rcs.items() if rc != 0}
        if failed and not args.continue_on_error:
            log(
                "[stop] non-zero exit in parallel round; use "
                "--continue-on-error to keep going."
            )
            stop_everything = True

    log("\n========== SUMMARY ==========")
    n_ok = sum(1 for _, _, rc in results if rc == 0)
    for task, abl, rc in results:
        log(
            f"  {task:<8s}  {abl:<14s}  "
            f"{'OK' if rc == 0 else f'FAIL(rc={rc})'}"
        )
    log(f"Total: {n_ok}/{len(results)} succeeded")
    log(f"Full log: {session_log}")

    return 0 if all(rc == 0 for _, _, rc in results) else 1


if __name__ == "__main__":
    sys.exit(main())
