#!/usr/bin/env python3
r"""Run harness ablation experiments across 4 tasks x 5 configurations.

Ablation semantics (one-hot harness, plus an all-on baseline):
  - with_harness: h2=h3=h4=h5=true          (everything on)
  - with_h2:      only h2=true, others false (only h2 on)
  - with_h3:      only h3=true, others false
  - with_h4:      only h4=true, others false
  - with_h5:      only h5=true, others false

For every (task, ablation) pair the script will:
  1. Patch configs/tasks/<task>.yaml so that the <task>-std: profile has
     the right h2/h3/h4/h5 flags (other profiles -- default, <task>-env_train
     -- are never touched).
  2. Patch configs/assignments/<task>.yaml so that exactly one output:
     line is uncommented, matching the current ablation.
  3. docker compose -f extra/docker-compose.yml up -d --force-recreate
     redis controller <task>-std.
  4. python -m src.assigner --config configs/assignments/<task>.yaml.

Only the -std (test) profile is exercised; -env_train profiles are left alone.

Targeting logic
---------------
The script targets the <task>-std: profile only by using a small state
machine in edit_task_yaml():
  * Top-level profile headers in these yaml files are flush against
    column 0 and end with ':' (e.g. "default:", "alfworld-std:",
    "alfworld-env_train:").
  * A flag "in_profile" turns on when we see a line whose stripped value
    equals "<task>-std:" and which is NOT indented.
  * It turns off again the next time we hit a top-level header.
  * Inside the profile (and only there) every line matching
    ^\s*(h[2345]):\s*(true|false)\s*$ has its value rewritten according
    to the flags dict. Everything else -- comments, h5_top_k, enabled,
    other nested keys -- is copied verbatim.

For dbbench the h-flags live under harness: with deeper indent; the regex
is indentation-agnostic, so it still hits only lines inside the current
profile's harness block.

Usage
-----

    cd my_agent_bench
    python run_ablations.py                       # run everything (4 x 5 = 20)
    python run_ablations.py --tasks alfworld      # only alfworld, all 5
    python run_ablations.py --ablations with_h2 with_harness
    python run_ablations.py --dry-run             # preview without touching
                                                  # files or running commands
    python run_ablations.py --only-edit           # patch yamls only; no runs
    python run_ablations.py --skip-docker         # assume services are up
    python run_ablations.py --continue-on-error   # keep going past failures
    python run_ablations.py --ready-timeout 0     # skip controller readiness wait
    python run_ablations.py --ready-timeout 600   # give slow tasks more time

Readiness check
---------------
After `docker compose up`, the <task>-std worker container still needs to
load its dataset before it registers itself with the controller. If we fire
the assigner too early we get::

    AgentBenchException: ('{"message":"task <task>-std does not exist"}', 400, ...)

To avoid that race, the script polls the controller's GET /get_indices
endpoint for <task>-std and only proceeds once it answers 200. The default
timeout is 300s (override with --ready-timeout; set 0 to disable).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "configs" / "tasks"
ASSIGN_DIR = ROOT / "configs" / "assignments"
COMPOSE_FILE = ROOT / "extra" / "docker-compose.yml"
LOG_DIR = ROOT / "ablation_logs"
CONTROLLER_URL = "http://localhost:5020/api"

TASKS: List[str] = ["alfworld", "dbbench", "webshop", "os"]
ABLATIONS: List[str] = ["with_harness", "with_h2", "with_h3", "with_h4", "with_h5"]


def flags_for(ablation: str) -> Dict[str, bool]:
    """Translate an ablation label into the h2/h3/h4/h5 boolean dict.

    - 'with_harness'  -> everything True
    - 'with_h<N>'     -> only h<N> True, the other three False
    """
    keys = ("h2", "h3", "h4", "h5")
    if ablation == "with_harness":
        return {k: True for k in keys}
    if not ablation.startswith("with_h"):
        raise ValueError(f"Unknown ablation label: {ablation!r}")
    on = ablation.split("_", 1)[1]  # e.g. "h2"
    if on not in keys:
        raise ValueError(f"Unknown harness name in ablation: {on!r}")
    return {k: (k == on) for k in keys}


def edit_task_yaml(path: Path, task: str, flags: Dict[str, bool]) -> None:
    """Set h2/h3/h4/h5 inside the <task>-std: profile only.

    Preserves every other line (comments, indentation, unrelated fields,
    the default: and <task>-env_train: profiles) verbatim.
    """
    target_header = f"{task}-std:"
    text = path.read_text()
    lines = text.split("\n")

    flag_re = re.compile(r"^(\s*)(h[2345]):\s*(true|false)\s*$", re.IGNORECASE)
    # A top-level profile header starts at column 0 and ends with ':'.
    top_level_re = re.compile(r"^\S[^:\s]*.*:\s*$")

    in_profile = False
    new_lines: List[str] = []
    for line in lines:
        if not in_profile:
            if line.strip() == target_header and not line.startswith((" ", "\t")):
                in_profile = True
            new_lines.append(line)
            continue

        # In-profile: exit as soon as another top-level header appears.
        if top_level_re.match(line) and not line.startswith((" ", "\t")):
            in_profile = False
            new_lines.append(line)
            continue

        m = flag_re.match(line)
        if m:
            indent, key, _ = m.groups()
            value = "true" if flags[key] else "false"
            new_lines.append(f"{indent}{key}: {value}")
        else:
            new_lines.append(line)

    path.write_text("\n".join(new_lines))


_OUTPUT_RE = re.compile(
    r'^(?P<comment>#\s*)?output:\s*["\'](?P<prefix>.+?)\{TIMESTAMP\}-(?P<suffix>[^"\']+)["\']\s*$'
)


def edit_assignment_yaml(path: Path, target_ablation: str) -> None:
    """Rewrite the block of output: lines so only `target_ablation` is active.

    Looks for a contiguous run of ``[#] output: "<prefix>{TIMESTAMP}-<suffix>"``
    lines, keeps the shared <prefix>, and regenerates exactly one line per
    ABLATION. The desired ablation's line is uncommented; the others are
    commented out with "# ".
    """
    text = path.read_text()
    lines = text.split("\n")

    indices: List[int] = []
    prefix: str | None = None
    for i, line in enumerate(lines):
        m = _OUTPUT_RE.match(line)
        if m:
            indices.append(i)
            prefix = m.group("prefix")  # e.g. "outputs/qwen3-4b-instruct/alfworld/"

    if not indices or prefix is None:
        raise RuntimeError(
            f"Could not locate an 'output: \"...{{TIMESTAMP}}-...\"' block in {path}"
        )

    new_block: List[str] = []
    for a in ABLATIONS:
        head = "" if a == target_ablation else "# "
        new_block.append(f'{head}output: "{prefix}{{TIMESTAMP}}-{a}"')

    start, end = indices[0], indices[-1]
    merged = lines[:start] + new_block + lines[end + 1 :]
    path.write_text("\n".join(merged))


def wait_for_task_registered(
    task_name: str,
    controller_url: str,
    timeout: float,
    interval: float,
    log: Callable[[str], None],
) -> bool:
    """Poll the controller until `task_name` is registered (or timeout).

    Uses the same endpoint the assigner calls: GET /get_indices?name=<task>.
    A 200 response means the task is registered and indices are ready; a 400
    typically means "task ... does not exist" (worker not up yet); connection
    errors mean the controller itself is still booting.
    """
    url = (
        controller_url.rstrip("/")
        + "/get_indices?"
        + urllib.parse.urlencode({"name": task_name})
    )
    deadline = time.monotonic() + timeout
    attempt = 0
    last_reason = "no attempts"
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    log(
                        f"[ready] task '{task_name}' registered with controller "
                        f"after {attempt} probe(s)"
                    )
                    return True
                last_reason = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            # 400 "task ... does not exist" is the expected transient state
            # while the worker is still loading its dataset.
            last_reason = f"HTTP {e.code}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_reason = f"conn: {e}"
        if attempt == 1 or attempt % 10 == 0:
            log(
                f"[wait] task '{task_name}' not ready yet "
                f"(attempt {attempt}, last={last_reason}); retrying..."
            )
        time.sleep(interval)
    log(
        f"[timeout] task '{task_name}' did not register within {timeout:.0f}s "
        f"(last={last_reason})"
    )
    return False


def run_cmd(
    cmd: List[str], cwd: Path, dry_run: bool, log_file: Path | None = None
) -> int:
    """Run `cmd` synchronously; tee stdout/stderr into console + log_file."""
    printable = " ".join(cmd)
    banner = f"$ (cwd={cwd}) {printable}"
    print(banner, flush=True)
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(banner + "\n")
    if dry_run:
        return 0

    if log_file is None:
        return subprocess.call(cmd, cwd=str(cwd))

    with log_file.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        proc.wait()
        return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run harness ablation experiments (4 tasks x 5 configs).",
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
        default=ABLATIONS,
        choices=ABLATIONS,
        help="Which ablations to run for each task (default: all 5).",
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
        help="Seconds to sleep after webshop docker up (webshop boots slowly).",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=300,
        help=(
            "Max seconds to wait for the <task>-std worker to register with the "
            "controller after 'docker compose up'. 0 disables the wait."
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
            "If set, write patched YAMLs into this directory instead of editing "
            "configs/tasks/*.yaml and configs/assignments/*.yaml. Implies no "
            "docker / assigner runs. Files are named "
            "<preview_dir>/<task>/<ablation>/{task,assignment}.yaml so you can "
            "inspect all 20 variants side-by-side. The originals stay untouched."
        ),
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_log = LOG_DIR / f"ablation-{session_id}.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with session_log.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    preview_root: Path | None = Path(args.preview_dir).resolve() if args.preview_dir else None

    log(f"# Ablation session {session_id}")
    log(f"# tasks       = {args.tasks}")
    log(f"# ablations   = {args.ablations}")
    log(f"# dry_run     = {args.dry_run}")
    log(f"# only_edit   = {args.only_edit}")
    log(f"# skip_docker = {args.skip_docker}")
    log(f"# preview_dir = {preview_root}")

    results: List[Tuple[str, str, int]] = []
    stop_everything = False

    for task in args.tasks:
        if stop_everything:
            break
        src_task_yaml = TASKS_DIR / f"{task}.yaml"
        src_assign_yaml = ASSIGN_DIR / f"{task}.yaml"
        if not src_task_yaml.exists() or not src_assign_yaml.exists():
            log(f"[skip] config missing for task={task}: {src_task_yaml} / {src_assign_yaml}")
            continue

        for abl in args.ablations:
            flags = flags_for(abl)
            header = (
                f"\n========== task={task}  ablation={abl}  "
                f"flags={{h2:{flags['h2']}, h3:{flags['h3']}, "
                f"h4:{flags['h4']}, h5:{flags['h5']}}}  =========="
            )
            log(header)

            # Decide where the patched YAMLs go.
            # - preview_dir set  -> write to <preview_dir>/<task>/<ablation>/,
            #                       never touch the originals.
            # - otherwise        -> patch the originals in-place.
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
                edit_task_yaml(task_yaml, task, flags)
                edit_assignment_yaml(assign_yaml, abl)
            log(f"[patched] {task_yaml}")
            log(f"[patched] {assign_yaml}")

            # Preview mode never runs docker / assigner.
            if preview_root is not None:
                results.append((task, abl, 0))
                continue

            if args.only_edit:
                results.append((task, abl, 0))
                continue

            if not args.skip_docker:
                rc = run_cmd(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(COMPOSE_FILE),
                        "up",
                        "-d",
                        "--force-recreate",
                        "redis",
                        "controller",
                        f"{task}-std",
                    ],
                    cwd=ROOT,
                    dry_run=args.dry_run,
                    log_file=session_log,
                )
                if rc != 0:
                    log(f"[error] docker compose up failed (rc={rc}) for task={task}")
                    results.append((task, abl, rc))
                    if not args.continue_on_error:
                        stop_everything = True
                        break
                    continue

                if task == "webshop" and args.webshop_wait > 0 and not args.dry_run:
                    log(
                        f"[wait] webshop needs extra boot time; sleeping "
                        f"{args.webshop_wait}s..."
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
                    results.append((task, abl, 124))
                    if not args.continue_on_error:
                        stop_everything = True
                        break
                    continue

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
            results.append((task, abl, rc))

            if rc != 0 and not args.continue_on_error:
                log("[stop] non-zero exit; use --continue-on-error to keep going.")
                stop_everything = True
                break

    log("\n========== SUMMARY ==========")
    n_ok = sum(1 for _, _, rc in results if rc == 0)
    for task, abl, rc in results:
        log(f"  {task:<8s}  {abl:<14s}  {'OK' if rc == 0 else f'FAIL(rc={rc})'}")
    log(f"Total: {n_ok}/{len(results)} succeeded")
    log(f"Full log: {session_log}")

    return 0 if all(rc == 0 for _, _, rc in results) else 1


if __name__ == "__main__":
    sys.exit(main())
