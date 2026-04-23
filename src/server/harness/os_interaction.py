"""
OS Interaction Harness — H0/H1/H2/H3/H4/H5  (v5)

H0  Task Parser            — one-time shell-task classification per episode
H1  Shell & Conv State     — per-round bash history, output candidates, truncation flags
H2  Action Gate            — text-embedded tool-call rescue (JSON/kwarg/positional/bare/XML)
                             + safety filter + duplicate-bash gate
                             + answer normalisation (strip units / prose, decimal sizes)
                               applied at rescue time AND commit time.
H3  Tool Description Patch — static shell strategy / tool-format hints
H4  Post-step Monitor      — truncation / error / empty-output / loop + budget warn/force
                             (budget logic is part of H4 — single post-bash trigger)
H5  Goal-directed Hint     — task-type-aware skill (BM25) + per-step guidance
                             (lint only fires when no candidate exists to avoid over-correction;
                              submit hint suppressed when H4 has active recovery prompt)

Architecture changes (v5 vs v4):
  - H5 round-0 submit hint changed to verification nudge: "you got 'N' on your first
    bash command — verify your command was exactly right before submitting."  Previously
    the immediate "Hint: submit 'N'" caused agents to submit 1-shot wrong answers
    (22/35 fails in v4 analysis).  From round 1+, the original prompt-to-submit resumes.
  - H4 post-ls nudge: after H4 fires recovery_prompt (path error / empty), if the NEXT
    bash is `ls` and shows files, inject "files confirmed — run your count command now."
    Fixes agents that confirmed file existence via ls but then submitted 0 anyway.
  - H4 grep-c output detection: when bash output contains multiple lines of `file:N` format
    (grep -c per-file output), warn and suggest the correct summation approach.
  - H5 new skill `date_pattern_extract`: warns against literal 'YYYY-MM-DD' in grep,
    shows correct `grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}'` pattern.
  - H5 new skill `lines_total_vs_excluding`: clarifies total vs containing vs excluding lines.
  - H5 `count_lines_matching` skill updated to mention -i flag for case-insensitive.
  - OSShellState gains `h4_fired_last_round` flag (used by post-ls nudge).

Architecture changes (v4 vs v3):
  - H5 submit hint now suppressed when H4 has an active recovery prompt (h4_audit_active
    flag passed from task.py).  Previously H4 and H5 emitted conflicting signals — H4
    said "retry, path is missing" while H5 said "submit '0'" — and the model followed H5.
  - Zero-candidate hint changed from unconditional "submit" to a verification nudge:
    "result is 0 — verify path/filter before submitting."  Reduces wrong-zero submissions.
  - String answer promotion extended from LARGEST/SMALLEST only to any task with a clean
    single-line output (answer_shape=ANSWER_STRING or None).  Fixes tasks like "find the
    date with most events" where the model found the answer but couldn't submit it.
  - H4 error detection broadened: "paths must precede expression" and "unary operator"
    now trigger no_such_file recovery, catching bash pipe mistakes.
  - no_such_file recovery prompt now explicitly warns: "the '0' may be from an error exit,
    not a real count — DO NOT submit it."
  - XML rescue feedback: when <tool_call> found in content but JSON parse fails (truncated),
    task.py injects an explicit feedback message so the model stops looping.
  - count_lines_matching skill updated to warn explicitly against grep -c | wc -l antipattern.
  - New skills: home_dir_file_glob (glob pattern fix), ip_status_extract (awk IP+status).

Architecture changes (v3 vs v2):
  - Budget management is a sub-branch of H4: post_step_monitor(remaining_rounds)
    handles budget warn/force alongside stall/error/empty checks.
  - Answer normalisation runs at H2 rescue time in addition to commit time.
  - Implausibility check restricted to answer_shape==ANSWER_INTEGER.
  - Zero is no longer unconditionally implausible.

Generalization notes:
  - H0 parsing is commonsense regex/lexical; no training-set statistics.
  - Skill library (OS_SKILLS) ranks by BM25 against the episode's raw description.
  - H4 budget force only fires when H1 has a concrete plausible integer candidate.
  - H2 answer normaliser is shape-conditional: only mutates if H0 set answer_shape.
  - All thresholds are behavioural (round counts, repeat counts, char lengths).
"""

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Task / answer shape constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_COUNT_FILES    = "count_files"
TASK_COUNT_LINES    = "count_lines"
TASK_COUNT_MATCHES  = "count_matches"
TASK_COUNT_UNIQUE   = "count_unique"
TASK_LARGEST        = "largest"
TASK_SMALLEST       = "smallest"
TASK_LIST           = "list"
TASK_READ_CONTENT   = "read_content"
TASK_SYSTEM_INFO    = "system_info"
TASK_SUM_SIZE       = "sum_size"        # "total size of all .log files …"
TASK_AVERAGE        = "average"         # "average age of …"
TASK_MUTATE         = "mutate"
TASK_OTHER          = "other"

_ALL_TASK_TYPES: List[str] = [
    TASK_COUNT_FILES, TASK_COUNT_LINES, TASK_COUNT_MATCHES, TASK_COUNT_UNIQUE,
    TASK_LARGEST, TASK_SMALLEST, TASK_LIST, TASK_READ_CONTENT,
    TASK_SYSTEM_INFO, TASK_SUM_SIZE, TASK_AVERAGE,
    TASK_MUTATE, TASK_OTHER,
]

ANSWER_INTEGER = "integer"
ANSWER_STRING  = "string"
ANSWER_SIZE    = "size"
ANSWER_PATH    = "path"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OSHarnessConfig:
    enabled: bool = False
    h2_enabled: bool = True
    h3_enabled: bool = True
    h4_enabled: bool = True   # post-step monitor + budget warn/force
    h5_enabled: bool = True

    # H2
    h2_repeat_bash_block_after: int = 2     # same bash N times → force answer
    h2_text_only_streak_force: int = 2      # N consecutive text-only turns → hard force

    # H4
    h4_stall_window: int = 3                # turns of same bash+output → stall
    h4_empty_output_threshold: int = 2      # empty outputs in a row → broaden-filter hint

    # H4-E hint
    h4_hint_max_words: int = 40

    # H5
    h5_top_k: int = 2
    h5_cold_start_max_words: int = 50

    # H4 budget (thresholds used by post_step_monitor ⑦)
    h4_budget_warn_threshold: int = 3       # remaining <= N → soft warn
    h4_budget_force_threshold: int = 2      # remaining <= N + candidate ready → hard force

    # H2 answer normalisation
    h2_max_strip_units: int = 1             # how many trailing unit tokens to strip


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retrieval (for H5 cold-start)
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _skill_doc_tokens(skill: Dict[str, Any]) -> List[str]:
    parts = [skill.get("text", "")] + list(skill.get("keywords", []))
    return _bm25_tokenize(" ".join(parts))


def _bm25_scores(
    query_tokens: List[str],
    docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    if not docs:
        return []
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / n_docs if n_docs else 1
    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    scores: List[float] = []
    unique_q = list(dict.fromkeys(query_tokens))
    for doc in docs:
        dl = len(doc)
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in unique_q:
            if t not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            f = tf[t]
            denom = f + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * f * (k1 + 1.0) / denom
        scores.append(score)
    return scores


def _truncate_to_word_budget(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Skill library (H5 cold-start)
# ─────────────────────────────────────────────────────────────────────────────

OS_SKILLS: List[Dict[str, Any]] = [
    {
        "id": "count_files_by_ext",
        "task_types": [TASK_COUNT_FILES],
        "keywords": ["count", "files", "extension", "txt", "log", "py", "csv", "json"],
        "text": (
            "To count files with a given extension, use "
            "`find DIR -type f -name '*.EXT' | wc -l`. "
            "`-type f` excludes directories — essential when the task says count files."
        ),
    },
    {
        "id": "count_files_mtime",
        "task_types": [TASK_COUNT_FILES],
        "keywords": ["modified", "last", "days", "hours", "recent", "recently", "mtime"],
        "text": (
            "For 'modified in the last N days', use "
            "`find DIR -type f -mtime -N | wc -l`. "
            "`-mtime -N` means within N days; `-mtime N` means exactly N days old."
        ),
    },
    {
        "id": "count_files_size",
        "task_types": [TASK_COUNT_FILES],
        "keywords": ["size", "greater", "larger", "smaller", "kb", "mb", "bytes"],
        "text": (
            "For file-size filters, use `-size +NK` (larger than N KB), `-size +NM` "
            "(larger than N MB), or `-size -NK` (smaller). Example: "
            "`find DIR -type f -size +100k | wc -l`."
        ),
    },
    {
        "id": "count_subdirs",
        "task_types": [TASK_COUNT_FILES],
        "keywords": ["directory", "directories", "subdirectory", "subdirectories", "folders"],
        "text": (
            "To count subdirectories (excluding the parent itself), use "
            "`find DIR -mindepth 1 -type d | wc -l`. Swap `-type d` for `-type f` "
            "to count files instead."
        ),
    },
    {
        "id": "recursion_hint",
        "task_types": [TASK_COUNT_FILES, TASK_COUNT_LINES, TASK_COUNT_MATCHES, TASK_COUNT_UNIQUE],
        "keywords": ["subdirectories", "recursively", "including", "nested"],
        "text": (
            "If the task mentions 'subdirectories' or 'recursively', use `find` "
            "(always recursive by default) or `grep -r`. Plain `ls` and `grep` "
            "without `-r` do NOT descend into subdirectories."
        ),
    },
    {
        "id": "exclude_dirs_hint",
        "task_types": [TASK_COUNT_FILES],
        "keywords": ["files", "exclude", "only", "regular"],
        "text": (
            "When counting *files* (not directories), always add `-type f` to `find`. "
            "Without it, directories are counted too and the answer is wrong."
        ),
    },
    {
        "id": "count_lines_matching",
        "task_types": [TASK_COUNT_MATCHES, TASK_COUNT_LINES],
        "keywords": ["lines", "containing", "matching", "word", "pattern", "grep"],
        "text": (
            "To count lines containing a pattern: `grep -rh PATTERN DIR --include='*.EXT' | wc -l`. "
            "For case-insensitive, add -i: `grep -rhi PATTERN DIR | wc -l`. "
            "NEVER pipe `grep -c FILE` to `wc -l` — `grep -c` returns count *per file*; "
            "`wc -l` then counts files, not total matching lines."
        ),
    },
    {
        "id": "count_lines_total",
        "task_types": [TASK_COUNT_LINES],
        "keywords": ["total", "lines", "all", "file", "wc"],
        "text": (
            "For total line count: `wc -l < FILE` (single file) or "
            "`find DIR -type f -name '*.log' -exec cat {} + | wc -l` for many files."
        ),
    },
    {
        "id": "count_unique_field",
        "task_types": [TASK_COUNT_UNIQUE],
        "keywords": ["unique", "distinct", "different", "ip", "user", "token", "address"],
        "text": (
            "To count unique values: extract them first (grep -oE or awk), then "
            "`sort -u | wc -l`. Example for unique IPs: "
            "`grep -oE '([0-9]+\\.){3}[0-9]+' FILE | sort -u | wc -l`."
        ),
    },
    {
        "id": "case_insensitive_hint",
        "task_types": [TASK_COUNT_MATCHES, TASK_COUNT_FILES, TASK_COUNT_UNIQUE, TASK_COUNT_LINES],
        "keywords": ["ignoring", "case", "regardless", "case-insensitive", "insensitive"],
        "text": (
            "Task says 'ignoring case' or 'regardless of case' → add `-i` to "
            "grep (`grep -i`) or use `find ... -iname` instead of `-name`."
        ),
    },
    {
        "id": "find_largest",
        "task_types": [TASK_LARGEST],
        "keywords": ["largest", "biggest", "max", "most"],
        "text": (
            "To find the largest file: "
            "`find DIR -type f -printf '%s %p\\n' | sort -rn | head -1`. "
            "For the filename only, pipe through `awk '{print $2}'`."
        ),
    },
    {
        "id": "find_smallest",
        "task_types": [TASK_SMALLEST],
        "keywords": ["smallest", "min", "least", "lowest"],
        "text": (
            "To find the smallest non-empty file: "
            "`find DIR -type f -size +0 -printf '%s %p\\n' | sort -n | head -1`."
        ),
    },
    {
        "id": "sum_sizes",
        "task_types": [TASK_SYSTEM_INFO, TASK_COUNT_FILES],
        "keywords": ["total", "size", "disk", "usage", "du"],
        "text": (
            "Total disk usage of a dir: `du -sh DIR` (human readable) or "
            "`du -sb DIR` (bytes). For matching files only: "
            "`find DIR -type f -name '*.EXT' -exec du -b {} + | awk '{s+=$1}END{print s}'`."
        ),
    },
    {
        "id": "system_disk_free",
        "task_types": [TASK_SYSTEM_INFO],
        "keywords": ["disk", "free", "available", "space", "df"],
        "text": "Disk free / used: `df -h /` or `df -h DIR`. Columns are Size, Used, Avail, Use%.",
    },
    {
        "id": "system_mem",
        "task_types": [TASK_SYSTEM_INFO],
        "keywords": ["memory", "ram", "free", "used", "mb", "gb"],
        "text": "Memory: `free -m` (MB) or `free -h` (human). `cat /proc/meminfo` for raw.",
    },
    {
        "id": "process_count",
        "task_types": [TASK_SYSTEM_INFO, TASK_COUNT_MATCHES],
        "keywords": ["process", "processes", "running", "pid", "ps"],
        "text": (
            "Process count: `ps -e --no-headers | wc -l`. For a specific name: "
            "`pgrep -c NAME` or `ps aux | grep -c '[N]AME'`."
        ),
    },
    {
        "id": "answer_format_reminder",
        "task_types": _ALL_TASK_TYPES,
        "keywords": ["answer", "format", "number", "count", "output"],
        "text": (
            "Submit only the bare value (e.g. '5', not '5 files' or 'The answer is 5'). "
            "Counting tasks expect a plain integer."
        ),
    },
    {
        "id": "no_repeat_when_done",
        "task_types": _ALL_TASK_TYPES,
        "keywords": ["already", "have", "answer", "submit", "result"],
        "text": (
            "If the last bash output already contains your answer, do NOT re-run "
            "the same command — submit your final answer immediately."
        ),
    },
    {
        "id": "truncation_handling",
        "task_types": _ALL_TASK_TYPES,
        "keywords": ["truncate", "truncated", "long", "output", "large"],
        "text": (
            "If output is truncated, refine the command — add `| wc -l`, "
            "`| head -20`, or a narrower filter. Do NOT re-run the same command."
        ),
    },
    {
        "id": "grep_bracket_word_bug",
        "task_types": _ALL_TASK_TYPES,
        "keywords": ["error", "warning", "info", "debug", "pattern", "match", "word", "contain"],
        "text": (
            "Regex pitfall: `grep '[ERROR]'` is a character class matching E, R, or O — "
            "NOT the word 'ERROR'. To match the literal string use `grep 'ERROR'` (no brackets). "
            "Same applies to [WARNING], [INFO], [DEBUG] etc."
        ),
    },
    {
        "id": "today_date_command",
        "task_types": [TASK_COUNT_MATCHES, TASK_COUNT_LINES, TASK_COUNT_UNIQUE, TASK_COUNT_FILES],
        "keywords": ["today", "current date", "this day", "date"],
        "text": (
            "When the task says 'today', get the actual date with `date +%Y-%m-%d` and use it in "
            "your grep: `grep \"$(date +%Y-%m-%d)\" FILE | ...`. "
            "Do NOT hardcode a date from the log file — that may be the wrong day."
        ),
    },
    # ── v2 specialised skills (added after 2026-04-18 eval) ──────────────────
    {
        "id": "human_readable_total_size",
        "task_types": [TASK_SUM_SIZE, TASK_SYSTEM_INFO],
        "keywords": ["total", "size", "sum", "human-readable", "kb", "mb", "gb", "combined"],
        "text": (
            "For 'total size … human-readable', use `du -ch $(find DIR -type f -name '*.EXT') "
            "| tail -1 | awk '{print $1}'` — output is '50K' (integer, no decimal). "
            "Do NOT use `printf \"%.1fK\"` — the evaluator needs int() parseable values."
        ),
    },
    {
        "id": "grep_time_window",
        "task_types": [TASK_COUNT_UNIQUE, TASK_COUNT_MATCHES, TASK_COUNT_LINES],
        "keywords": ["between", "time", "hour", "window", "from", "timestamp", "inclusive"],
        "text": (
            "For an HH window (e.g. 02:00–04:00 inclusive), use "
            "`grep -hE '^(02|03|04):' FILES`, NOT an enumerated minute list.  "
            "For HH:MM sub-ranges use `awk -F'[: ]' '$1>=2 && $1<=4'`.  "
            "Strip the timestamp field before `sort -u | wc -l`."
        ),
    },
    {
        "id": "awk_safe_average",
        "task_types": [TASK_AVERAGE, TASK_SYSTEM_INFO, TASK_OTHER],
        "keywords": ["average", "mean", "round", "integer", "compute", "age"],
        "text": (
            "Safe average with zero-division guard: "
            "`awk -F: '$2 ~ /^[0-9]+$/ {s+=$2; c++} "
            "END{if(c>0) printf \"%.0f\", s/c; else print 0}' FILE`.  "
            "Always pre-filter the numeric field with `/^[0-9]+$/` and guard `c>0`."
        ),
    },
    {
        "id": "files_containing_pattern",
        "task_types": [TASK_COUNT_FILES, TASK_COUNT_MATCHES],
        "keywords": ["files", "containing", "include", "with", "mention", "word", "which"],
        "text": (
            "Count FILES (not lines) that contain a pattern: "
            "`grep -rl PATTERN DIR | wc -l` (`-l` = files-with-matches).  "
            "Counting lines `grep -rh | wc -l` is different — pick by what the "
            "task asks for ('how many files' vs 'how many lines')."
        ),
    },
    {
        "id": "hidden_files",
        "task_types": [TASK_COUNT_FILES, TASK_LIST, TASK_READ_CONTENT],
        "keywords": ["hidden", "dotfile", "starts", "dot", "user_data", "bashrc"],
        "text": (
            "Hidden files begin with `.` and are skipped by default `ls`.  Use "
            "`ls -A DIR` (excludes . and ..) or `find DIR -maxdepth 1 -name '.*' -type f`.  "
            "To read `~/.user_data` use the literal `~/.user_data`, not `cat .user_data` "
            "unless you're already in $HOME."
        ),
    },
    {
        "id": "extract_field_then_unique",
        "task_types": [TASK_COUNT_UNIQUE, TASK_AVERAGE],
        "keywords": ["field", "column", "separator", "delimited", "colon", "csv", "tsv"],
        "text": (
            "For colon/CSV delimited data (`user:age:email`), extract one field: "
            "`awk -F: '{print $2}' FILE | sort -u | wc -l` for count_unique; "
            "`cut -d':' -f2 FILE | …` for simpler cases.  Always confirm the "
            "separator with `head -3 FILE` first when the format isn't obvious."
        ),
    },
    {
        "id": "count_nonrecursive",
        "task_types": [TASK_COUNT_FILES],
        "keywords": ["only", "top", "current", "directly", "immediately", "not", "subdirectories"],
        "text": (
            "'Only in the top directory' / 'not in subdirectories' → use "
            "`find DIR -maxdepth 1 -type f | wc -l`.  Plain `find DIR` recurses "
            "by default; `-maxdepth 1` keeps it shallow."
        ),
    },
    # ── v5 specialised skills (added after 2026-04-19 v4 eval) ──────────────
    {
        "id": "lines_total_vs_excluding",
        "task_types": [TASK_COUNT_LINES, TASK_COUNT_MATCHES],
        "keywords": ["excluding", "except", "without", "not containing", "total", "all lines", "non-empty"],
        "text": (
            "Total ALL lines: `wc -l FILE` or `find DIR -name '*.EXT' -exec cat {} + | wc -l`. "
            "Lines CONTAINING pattern: `grep -rh PATTERN DIR | wc -l`. "
            "Lines EXCLUDING pattern: `grep -v PATTERN FILE | wc -l` (add -r for dirs). "
            "Make sure to use grep -v (invert), NOT grep PATTERN, when task says 'excluding'."
        ),
    },
    {
        "id": "date_pattern_extract",
        "task_types": [TASK_COUNT_UNIQUE, TASK_COUNT_MATCHES, TASK_COUNT_LINES],
        "keywords": ["date", "dates", "timestamp", "unique", "year", "day", "month", "yyyy", "log", "entry"],
        "text": (
            "To extract unique dates from log files: "
            "`grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' ~/logs/*.log | sort -u | wc -l`. "
            "NEVER grep for literal 'YYYY-MM-DD' or 'MM/DD/YYYY' — those are templates, "
            "not real dates. Run `head -3 FILE` first to see the actual date format."
        ),
    },
    # ── v4 specialised skills (added after 2026-04-19 eval) ──────────────────
    {
        "id": "home_dir_file_glob",
        "task_types": [TASK_COUNT_LINES, TASK_COUNT_MATCHES, TASK_COUNT_FILES, TASK_COUNT_UNIQUE],
        "keywords": ["home", "txt", "text", "files", "home directory", "glob"],
        "text": (
            "Never use `~/.*\\.EXT` glob in bash — it produces 'No such file or directory'. "
            "Use `find DIR -name '*.EXT'` instead, where DIR is the target path. "
            "If the task specifies a subdirectory (e.g. '~/project_files'), use that full path — "
            "NOT just `~` or `find ~ -maxdepth 1`."
        ),
    },
    {
        "id": "ip_status_extract",
        "task_types": [TASK_COUNT_UNIQUE],
        "keywords": ["access", "http", "apache", "nginx", "web", "status", "200", "404", "request"],
        "text": (
            "For HTTP access logs: count unique IPs with status code N: "
            "`awk '$9==\"200\"{print $1}' access.log | sort -u | wc -l`. "
            "For SSH/auth logs: `grep 'Failed password' /var/log/auth.log | awk '{print $11}' | sort -u | wc -l`. "
            "Do NOT grep for the status code after extracting IPs — the status code is in a different field."
        ),
    },
]


def retrieve_os_skills(
    task_type: str,
    query: str,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """Two-layer retrieval: task_type filter + BM25 ranking against query."""
    candidates = [s for s in OS_SKILLS if task_type in s.get("task_types", [])]
    if not candidates:
        # Fallback to the always-on skills (tagged with every type)
        candidates = [s for s in OS_SKILLS if _ALL_TASK_TYPES[0] in s.get("task_types", [])]
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates
    query_tokens = _bm25_tokenize(query)
    if not query_tokens:
        return candidates[:top_k]
    docs = [_skill_doc_tokens(s) for s in candidates]
    scores = _bm25_scores(query_tokens, docs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# H0 — Task Parser
# ─────────────────────────────────────────────────────────────────────────────

_EXT_RE    = re.compile(r"\.([a-z0-9]{1,6})\b", re.IGNORECASE)
_MTIME_RE  = re.compile(r"last\s+(\d+)\s+(day|days|hour|hours|week|weeks|minute|minutes)", re.IGNORECASE)
_SIZE_RE   = re.compile(r"(?:greater|more|larger|bigger|exceed(?:s|ing)?)\s+than\s+(\d+)\s*(bytes?|kb|mb|gb|k|m|g)\b", re.IGNORECASE)
_SIZE_LT_RE = re.compile(r"(?:less|smaller)\s+than\s+(\d+)\s*(bytes?|kb|mb|gb|k|m|g)\b", re.IGNORECASE)
_PATH_RE   = re.compile(r"(?:(~/|/)[A-Za-z0-9_\-./]+)")
_DIR_NAME_RE = re.compile(r'"([A-Za-z0-9_\-]+)"(?:\s+(?:directory|folder|dir))', re.IGNORECASE)
_HOME_DIR_RE = re.compile(r'(?:named|called)\s+"([A-Za-z0-9_\-]+)"\s+in\s+(?:your\s+)?home', re.IGNORECASE)

_RECURSIVE_SIGNALS = [
    "subdirectories", "sub-directories", "subdirectory",
    "recursively", "recursive", "including subdirectories",
    "and its subdirectories", "nested",
]

_CASE_INSENS_SIGNALS = [
    "ignoring case", "regardless of case", "case-insensitive",
    "case insensitive", "irrespective of case",
]

_MUTATION_VERBS = [
    "create", "delete", "remove", "rename", "move", "copy",
    "change the permission", "chmod", "chown", "mkdir", "touch a file",
    "set the permission", "grant", "revoke", "make the file",
]


@dataclass
class OSTaskContext:
    raw_description: str = ""
    task_type: str = TASK_OTHER
    answer_shape: Optional[str] = None
    target_path: Optional[str] = None
    extension_filter: Optional[str] = None
    recursive: Optional[bool] = None
    case_sensitive: Optional[bool] = None
    time_filter_days: Optional[int] = None
    size_filter_bytes_gt: Optional[int] = None
    size_filter_bytes_lt: Optional[int] = None


def _normalize_size(value: int, unit: str) -> int:
    unit = unit.lower().rstrip("s")
    if unit in ("byte", ""):
        return value
    if unit in ("k", "kb"):
        return value * 1024
    if unit in ("m", "mb"):
        return value * 1024 * 1024
    if unit in ("g", "gb"):
        return value * 1024 * 1024 * 1024
    return value


def _detect_task_type(desc: str) -> str:
    t = desc.lower()
    # Mutation first (explicit verb list)
    for v in _MUTATION_VERBS:
        if v in t:
            return TASK_MUTATE
    # Aggregate size — "total/combined/overall size of … files".  Must come
    # before the count-family regexes, otherwise "files" lexical hit drags it
    # into count_files and shape is mis-classified as integer.
    if re.search(r"\b(total|combined|overall|aggregate|sum\s+of(?:\s+the)?)\s+size\b", t) or \
       re.search(r"\b(determine|calculate|compute|find)\s+the\s+(?:total|combined|overall)\s+size\b", t):
        return TASK_SUM_SIZE
    # Average — "average age / mean X / compute the average".  Tighter than
    # bare `\baverage\b` to avoid prose uses ("on average, …"); requires the
    # word to precede a common quantitative noun or follow a computation verb.
    if re.search(
        r"\b(?:average|mean)\s+(?:age|value|size|length|count|number|score|"
        r"time|rate|price|weight|height|duration|latency|salary|temperature)\b",
        t,
    ) or re.search(
        r"\b(?:compute|calculate|determine|find|get)\s+the\s+(?:average|mean|arithmetic\s+mean)\b",
        t,
    ):
        return TASK_AVERAGE
    # System info
    if re.search(r"\b(disk\s+(free|usage|space)|memory\s+(usage|free|available)|cpu\s+info|uptime|kernel)\b", t):
        return TASK_SYSTEM_INFO
    if re.search(r"\b(processes?|running\s+process)\b", t) and not re.search(r"\bcount\s+the\s+lines", t):
        return TASK_SYSTEM_INFO
    # Largest / smallest
    if re.search(r"\b(largest|biggest|max(?:imum)?)\b.*\bfile\b", t):
        return TASK_LARGEST
    if re.search(r"\b(smallest|min(?:imum)?)\b.*\bfile\b", t):
        return TASK_SMALLEST
    # Counting — check content first so 'count lines/matches' wins over generic 'count files'
    if re.search(r"\bunique\b", t) and re.search(r"\bcount|number\s+of|how\s+many\b", t):
        return TASK_COUNT_UNIQUE
    if re.search(r"\bcount\s+(?:the\s+)?(?:number\s+of\s+)?lines?\b", t) or \
       re.search(r"\bhow\s+many\s+lines\b", t) or \
       re.search(r"\btotal\s+number\s+of\s+lines\b", t):
        if re.search(r"\bcontain(?:ing)?\b|\bmatch(?:ing)?\b|\bwith\s+the\s+word\b", t):
            return TASK_COUNT_MATCHES
        return TASK_COUNT_LINES
    if re.search(r"\bcontain(?:ing)?\b|\bmatch(?:ing)?\b", t) and re.search(r"\bcount|number\s+of|how\s+many\b", t):
        return TASK_COUNT_MATCHES
    if re.search(r"\bcount\b|\bnumber\s+of\b|\bhow\s+many\b", t) and re.search(r"\bfiles?\b|\bdirector(?:y|ies)\b|\bfolders?\b", t):
        # "count entries/occurrences/errors in files" → TASK_COUNT_MATCHES, not TASK_COUNT_FILES
        if re.search(
            r"\bentries?\b|\boccurrences?\b|\binstances?\b|\blines?\b"
            r"|\berrors?\b|\bwarnings?\b|\blevel\b|\blog\s+entries?\b",
            t,
        ):
            return TASK_COUNT_MATCHES
        return TASK_COUNT_FILES
    # List / read
    if re.search(r"\blist\b.*\bfiles?\b", t):
        return TASK_LIST
    if re.search(r"\b(content|read|print|show)\b.*\bfile\b", t):
        return TASK_READ_CONTENT
    return TASK_OTHER


def _detect_answer_shape(desc: str, task_type: str) -> Optional[str]:
    t = desc.lower()
    if task_type in (TASK_COUNT_FILES, TASK_COUNT_LINES, TASK_COUNT_MATCHES, TASK_COUNT_UNIQUE):
        return ANSWER_INTEGER
    if task_type == TASK_AVERAGE:
        return ANSWER_INTEGER  # rounded integer is the common evaluator shape
    if task_type == TASK_SUM_SIZE:
        # Human-readable (KB/MB/GB) → size; pure bytes → integer.
        if re.search(r"\bhuman[-\s]?readable\b|\bkb\b|\bmb\b|\bgb\b|\bkilo|\bmega|\bgiga", t):
            return ANSWER_SIZE
        if re.search(r"\bin\s+bytes\b|\braw\s+bytes\b", t):
            return ANSWER_INTEGER
        return ANSWER_SIZE
    if task_type in (TASK_LARGEST, TASK_SMALLEST):
        # Could be filename or size — look for "filename" / "path" / "name"
        if re.search(r"\b(name|filename|path)\b", t):
            return ANSWER_STRING
        if re.search(r"\bsize\b", t):
            return ANSWER_SIZE
        return ANSWER_STRING
    if task_type == TASK_SYSTEM_INFO:
        if re.search(r"\bnumber\s+of|count|how\s+many\b", t):
            return ANSWER_INTEGER
        if re.search(r"\b(kb|mb|gb|bytes)\b", t):
            return ANSWER_SIZE
        return None
    if task_type == TASK_MUTATE:
        return None  # finish_action expected
    return None


def _detect_target_path(desc: str) -> Optional[str]:
    t = desc
    # Explicit path
    m = _PATH_RE.search(t)
    if m:
        return m.group(0).rstrip(".,")
    # Home directory references
    if re.search(r"\bhome\s+director(?:y|ies)\b", t, re.IGNORECASE):
        # Look for a named sub-directory
        dm = _HOME_DIR_RE.search(t) or _DIR_NAME_RE.search(t)
        if dm:
            return f"~/{dm.group(1)}"
        return "~"
    dm = _DIR_NAME_RE.search(t)
    if dm:
        return f"~/{dm.group(1)}"
    return None


def _detect_extension(desc: str) -> Optional[str]:
    # Prefer quoted extensions like "\".txt\"" → txt
    m = re.search(r'"\s*\.([a-z0-9]{1,6})\s*"', desc, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\.([a-z0-9]{1,6})\s+(?:extension|files?)\b", desc, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _detect_recursive(desc: str) -> Optional[bool]:
    t = desc.lower()
    for s in _RECURSIVE_SIGNALS:
        if s in t:
            return True
    if re.search(r"\bonly\s+in\s+(?:the\s+)?(?:top|current)\s+director(?:y|ies)\b", t):
        return False
    return None


def _detect_case_sensitivity(desc: str) -> Optional[bool]:
    t = desc.lower()
    for s in _CASE_INSENS_SIGNALS:
        if s in t:
            return False
    if "case-sensitive" in t or "case sensitive" in t:
        return True
    return None


def parse_task_context(description: str) -> OSTaskContext:
    ctx = OSTaskContext(raw_description=description or "")
    ctx.task_type = _detect_task_type(description or "")
    ctx.answer_shape = _detect_answer_shape(description or "", ctx.task_type)
    ctx.target_path = _detect_target_path(description or "")
    ctx.extension_filter = _detect_extension(description or "")
    ctx.recursive = _detect_recursive(description or "")
    ctx.case_sensitive = _detect_case_sensitivity(description or "")
    # Time filter
    mt = _MTIME_RE.search(description or "")
    if mt:
        n = int(mt.group(1))
        unit = mt.group(2).lower()
        if unit.startswith("day"):
            ctx.time_filter_days = n
        elif unit.startswith("week"):
            ctx.time_filter_days = n * 7
        elif unit.startswith("hour"):
            ctx.time_filter_days = max(1, n // 24)
    # Size filters
    sg = _SIZE_RE.search(description or "")
    if sg:
        ctx.size_filter_bytes_gt = _normalize_size(int(sg.group(1)), sg.group(2))
    sl = _SIZE_LT_RE.search(description or "")
    if sl:
        ctx.size_filter_bytes_lt = _normalize_size(int(sl.group(1)), sl.group(2))
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# H1 — Shell / conversation state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OSShellState:
    bash_history: List[str] = field(default_factory=list)
    last_output_raw: str = ""
    last_output_numeric_candidates: List[str] = field(default_factory=list)
    last_output_truncated: bool = False
    last_output_was_empty: bool = False
    last_output_had_error: bool = False
    empty_output_streak: int = 0
    text_only_streak: int = 0
    rescue_hits: int = 0
    answered: bool = False
    candidate_numeric_answer: Optional[str] = None
    candidate_string_answer: Optional[str] = None
    # H8 health: True when the candidate above looks garbage (awk overflow, 0
    # from an over-filtered set, negative, absurdly large).  Consumers use this
    # to refuse force-submit / force-promote.
    candidate_implausible: bool = False
    # v5: True when the previous round's post_step_monitor returned a recovery_prompt.
    # Used by the post-ls nudge to detect "H4 fired → agent ran ls → files found" pattern.
    h4_fired_last_round: bool = False
    # Cumulative count of bash_loop detections (resets per episode via dataclass init).
    bash_loop_count: int = 0


_NUMERIC_LINE_RE = re.compile(r"^\s*(-?\d+)\s*$")
_TRAILING_NUM_RE = re.compile(r"(-?\d+)\s*(?:total|files?|lines?|matches?)?\s*$", re.IGNORECASE)
# Detects grep -c per-file output: lines matching "path/file.ext:N"
_GREP_C_LINE_RE = re.compile(r"^[^\s:]+:\d+$")
_ERROR_RE = re.compile(
    r"(command not found|no such file or directory|permission denied|syntax error|"
    r"not a directory|paths must precede expression|unary operator expected|"
    r"binary file .{0,40} matches|cannot open|bad substitution|ambiguous redirect)",
    re.IGNORECASE,
)
_TRUNC_MARK = "[truncated because the output is too long]"

# awk/integer overflow sentinels that indicate a broken pipeline (count==0
# divisions, uninitialised sums, etc.) and should NEVER be force-submitted.
_AWK_OVERFLOW_SENTINELS = {
    "-9223372036854775808", "9223372036854775807", "nan", "inf", "-inf",
}


def is_plausible_numeric_candidate(value: Optional[str]) -> bool:
    """Conservative plausibility check for integer candidates.

    Returns False for values that almost always indicate the bash pipeline
    produced garbage (awk overflow, empty-set count 0 for filter tasks, huge
    negatives).  Used by H4 budget to refuse auto-submit / auto-force.
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    if not s:
        return False
    if s in _AWK_OVERFLOW_SENTINELS:
        return False
    if not re.fullmatch(r"-?\d+", s):
        # Non-integer string answers (size / filename) pass through — plausibility
        # is judged elsewhere.
        return True
    try:
        n = int(s)
    except Exception:
        return False
    if n < 0:
        return False
    if n > 10 ** 15:
        return False
    return True


def bash_semantic_gaps(ctx: "OSTaskContext", bash: str) -> List[str]:
    """Return a list of natural-language gaps between a bash command and the
    parsed task intent.  Used by H5 step_guidance as a light-weight lint so
    the agent gets a specific correction instead of repeating the same
    wrong pipeline.  Conservative — only reports high-confidence mismatches.
    """
    gaps: List[str] = []
    if not bash:
        return gaps
    b = bash.lower()
    # Extension filter missing
    if ctx.extension_filter and f".{ctx.extension_filter}" not in b \
            and f"*.{ctx.extension_filter}" not in b:
        gaps.append(
            f"your command didn't filter by `.{ctx.extension_filter}` — add "
            f"`-name '*.{ctx.extension_filter}'` (find) or restrict grep to `*.{ctx.extension_filter}`"
        )
    # Case-insensitive requested but not passed to the tool
    if ctx.case_sensitive is False:
        using_grep = "grep" in b
        using_find = re.search(r"\bfind\b", b) is not None
        has_i_flag = bool(re.search(r"grep\s+(?:-[a-zA-Z]*i[a-zA-Z]*|\S*-i\b)", b)) or "-i " in b
        has_iname = "-iname" in b
        if using_grep and not has_i_flag:
            gaps.append("task says 'ignoring case' — add `-i` to grep")
        if using_find and not has_iname and ctx.extension_filter:
            gaps.append("task says 'ignoring case' — use `-iname` instead of `-name`")
    # Recursion mismatch
    if ctx.recursive is True:
        recursive_tools = ("find", "grep -r", "grep -r ", "grep -rn", "grep -rh", "grep -rl", "grep -ri")
        if not any(t in b for t in recursive_tools):
            gaps.append("task mentions subdirectories — use `find` or `grep -r`, not plain `ls`/`grep`")
    if ctx.recursive is False and re.search(r"\bfind\b", b) and "-maxdepth" not in b:
        gaps.append("task says 'top directory only' — add `-maxdepth 1` to find")
    # -type f missing on count_files
    if ctx.task_type == TASK_COUNT_FILES and "find" in b and "-type f" not in b and "-type d" not in b:
        gaps.append("counting files — add `-type f` to exclude directories from the count")
    # [WORD] character-class bug: grep '[ERROR]' matches single chars E/R/O, not the word
    _bracket_word = re.search(r"grep\b[^|]*\[([A-Za-z]{2,})\]", bash)
    if _bracket_word:
        gaps.append(
            f"'[{_bracket_word.group(1)}]' is a regex character class, not the word "
            f"'{_bracket_word.group(1)}' — use `grep '{_bracket_word.group(1)}'` (no brackets)"
        )
    # xargs grep -r antipattern: -r makes grep recursive on file args from xargs — usually wrong
    if "xargs" in b and re.search(r"grep\s+(?:-[a-zA-Z]*r[a-zA-Z]*\s|.*\s-r\b)", bash):
        gaps.append(
            "avoid `xargs grep -r` — xargs passes filenames so `-r` recurses inside each file path; "
            "use `xargs grep` (no -r) or `grep -r PATTERN DIR` directly"
        )
    return gaps


def extract_numeric_candidates(text: str) -> List[str]:
    """Pull plausible integer answers out of a shell output.

    Priority:
      1. A single line that's just an integer.
      2. The last line matching `<N> total` (wc -l style).
      3. The very last integer on the last non-empty line.
    """
    if not text:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    out: List[str] = []
    # Rule 1: any line that is a pure integer
    for ln in lines:
        m = _NUMERIC_LINE_RE.match(ln)
        if m:
            out.append(m.group(1))
    if out:
        # Prefer the last pure-integer line
        return [out[-1]]
    # Rule 2: `<N> total`
    for ln in reversed(lines):
        tm = re.match(r"^\s*(-?\d+)\s+total\s*$", ln, re.IGNORECASE)
        if tm:
            return [tm.group(1)]
    # Rule 3: trailing integer on last line
    last = lines[-1]
    tm = _TRAILING_NUM_RE.search(last)
    if tm:
        return [tm.group(1)]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# H3 — Tool description patching
# ─────────────────────────────────────────────────────────────────────────────

_H3_BASH_HINT = (
    "Use targeted, readable commands — break complex logic into multiple turns. "
    "For counting files: `find DIR -type f -name '*.EXT' | wc -l` (always add `-type f`). "
    "For counting matching lines: `grep -rh PATTERN DIR | wc -l`. "
    "For unique values: pipe to `sort -u | wc -l`."
)

_H3_ANSWER_HINT = (
    "Return ONLY the bare value (e.g. '5', not '5 files' or 'The answer is 5'). "
    "Do NOT write `answer_action(...)` as plain text — you MUST invoke this tool "
    "via a real function call. Plain-text invocations are rejected."
)

_H3_FINISH_HINT = (
    "Use `finish_action` only for mutation tasks (create/delete/chmod/...). "
    "For a question with a numeric or string answer, use `answer_action`."
)


def patch_os_tool_descriptions(
    tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """H3: Append shell-strategy and format hints to tool descriptions."""
    if not tools:
        return tools
    patched = copy.deepcopy(tools)
    for tool in patched:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if name == "bash_action":
            fn["description"] = fn.get("description", "") + " " + _H3_BASH_HINT
        elif name == "answer_action":
            fn["description"] = fn.get("description", "") + " " + _H3_ANSWER_HINT
        elif name == "finish_action":
            fn["description"] = fn.get("description", "") + " " + _H3_FINISH_HINT
        tool["function"] = fn
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Rescue parser for text-embedded tool calls
# ─────────────────────────────────────────────────────────────────────────────

# JSON-dict style: answer_action({"answer":"5"}) or answer_action({"answer": "5"})
_RESCUE_ANSWER_JSON_RE = re.compile(
    r"answer_action\s*[\({]\s*\{?\s*[\"']?answer[\"']?\s*:\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
# Python-kwarg style: answer_action(answer='5') / answer_action(answer="5") — the
# dominant failure mode on Qwen3-4B-Instruct (38/43 fails in 2026-04-18 eval).
_RESCUE_ANSWER_KWARG_RE = re.compile(
    r"answer_action\s*\(\s*answer\s*=\s*(?:['\"]([^'\"]{1,200})['\"]|([^\)\n]{1,120}))\s*\)",
    re.IGNORECASE,
)
# Positional form: answer_action(5) / answer_action('42.5MB') / answer_action("foo")
_RESCUE_ANSWER_POSITIONAL_RE = re.compile(
    r"answer_action\s*\(\s*(?:['\"]([^'\"]{1,200})['\"]|([^\)\n]{1,120}))\s*\)",
    re.IGNORECASE,
)
# Bare "answer_action 60" (no parens) — only match a tight line-final token so
# prose like "call answer_action with the value" isn't rescued to garbage.
_RESCUE_ANSWER_BARE_RE = re.compile(
    r"(?m)^\s*answer_action[\s:=]+['\"]?([A-Za-z0-9_.+\-/]{1,60})['\"]?\s*$",
    re.IGNORECASE,
)
_RESCUE_BASH_RE = re.compile(
    r"bash_action\s*[\({]\s*\{?\s*[\"']?script[\"']?\s*:\s*[\"']((?:[^\"'\\]|\\.)+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_RESCUE_BASH_KWARG_RE = re.compile(
    r"bash_action\s*\(\s*script\s*=\s*['\"]((?:[^'\"\\]|\\.)+)['\"]",
    re.IGNORECASE | re.DOTALL,
)
_RESCUE_FINISH_RE = re.compile(
    r"finish_action\s*[\({]\s*\{?\s*[\"']?thought[\"']?\s*:\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_RESCUE_FINISH_KWARG_RE = re.compile(
    r"finish_action\s*\(\s*thought\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
# ReAct fallback: "Act: answer(5)" / "Act: bash" + ```bash ... ```
_REACT_ANSWER_RE = re.compile(r"Act:\s*answer\s*\(([^)]+)\)", re.IGNORECASE)
_REACT_BASH_FENCE = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)
# Qwen3 OpenAI-JSON style: <tool_call>\n{"name":"bash_action","arguments":{...}}\n</tool_call>
# MUST be greedy (.*) — the outer JSON has nested dicts (}} at the end), so non-greedy
# .*? stops at the first } (inner dict) and produces incomplete JSON that json.loads rejects.
_RESCUE_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
# Qwen3 writes bash grouping \( \) inside JSON strings, which is invalid JSON.
# Convert invalid \X to \\X so json.loads sees a valid escaped backslash, and
# the resulting parsed string retains \( as correct bash syntax.
_INVALID_JSON_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')


def _fix_json_escapes(s: str) -> str:
    """Convert invalid JSON escapes (e.g. backslash-paren) to double-backslash form."""
    return _INVALID_JSON_ESCAPE_RE.sub(r'\\\\\1', s)


def _first_nonempty_match_group(m: re.Match) -> str:
    """Return the first non-empty capture group from a regex match."""
    for g in m.groups():
        if g is not None and g != "":
            return g
    return m.group(0)


def rescue_tool_call_from_text(content: str) -> Optional[Dict[str, Any]]:
    """Attempt to lift a tool invocation from plain-text assistant content.

    Returns a dict like {"name": "answer_action", "arguments": {"answer": "5"}}
    or None if nothing recognisable was embedded.  Priority is answer > finish >
    bash — once the agent has written the answer in text we should submit it
    rather than re-run a command.  Answer patterns are tried in order from
    strictest (JSON dict) to loosest (bare line) to minimise false positives.
    """
    if not content:
        return None
    # XML <tool_call>{"name":...,"arguments":{...}}</tool_call> — Qwen3's OpenAI-JSON format.
    # Try this first because it's unambiguous (contains name + full arguments dict).
    m = _RESCUE_TOOL_CALL_XML_RE.search(content)
    if m:
        try:
            import json as _json
            obj = _json.loads(_fix_json_escapes(m.group(1)))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if name in ("answer_action", "bash_action", "finish_action") and args:
                return {"name": name, "arguments": args}
        except Exception:
            pass
    # answer_action — try strict → loose
    for pat in (_RESCUE_ANSWER_JSON_RE, _RESCUE_ANSWER_KWARG_RE,
                _RESCUE_ANSWER_POSITIONAL_RE, _RESCUE_ANSWER_BARE_RE):
        m = pat.search(content)
        if m:
            val = _first_nonempty_match_group(m).strip().strip("'\"")
            # Defend against picking up the literal token "answer" / "action"
            if val.lower() in {"answer", "action", "value"}:
                continue
            return {"name": "answer_action", "arguments": {"answer": val}}
    m = _REACT_ANSWER_RE.search(content)
    if m:
        val = m.group(1).strip().strip('"\'')
        return {"name": "answer_action", "arguments": {"answer": val}}
    # finish_action
    for pat in (_RESCUE_FINISH_RE, _RESCUE_FINISH_KWARG_RE):
        m = pat.search(content)
        if m:
            return {"name": "finish_action", "arguments": {"thought": m.group(1).strip()}}
    # bash_action
    for pat in (_RESCUE_BASH_RE, _RESCUE_BASH_KWARG_RE):
        m = pat.search(content)
        if m:
            script = m.group(1).encode().decode("unicode_escape", errors="ignore")
            return {"name": "bash_action", "arguments": {"script": script}}
    m = _REACT_BASH_FENCE.search(content)
    if m:
        return {"name": "bash_action", "arguments": {"script": m.group(1).strip()}}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Safety filter
# ─────────────────────────────────────────────────────────────────────────────

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/(?:\s|$)"),
    re.compile(r":\(\)\s*\{\s*:\|\s*:&\s*\};?:"),  # fork bomb
    re.compile(r"\bmkfs\."),
    re.compile(r"\bdd\s+.*of=/dev/"),
    re.compile(r">\s*/dev/(?:sda|sdb|nvme)"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bchmod\s+-R\s+777\s+/(?:\s|$)"),
]


def _is_dangerous_bash(script: str) -> Optional[str]:
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(script or ""):
            return pat.pattern
    return None


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Answer normaliser (interface-boundary repair at rescue / commit)
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_STRIP_PREFIXES = [
    "the answer is", "answer:", "answer is", "result:", "result is",
    "output:", "count:", "count is", "total:", "total is",
]

_ANSWER_UNIT_TOKENS = {
    "file", "files", "line", "lines", "byte", "bytes",
    "match", "matches", "occurrence", "occurrences",
    "directory", "directories", "folder", "folders",
    "process", "processes", "user", "users",
    "entry", "entries", "item", "items",
    "ip", "ips", "address", "addresses",
}


def normalize_answer(value: str, answer_shape: Optional[str]) -> Tuple[str, bool]:
    """H2: Normalise submitted answer based on H0's answer_shape.

    Returns (normalized, mutated). Passthrough when shape is None.
    """
    if value is None or answer_shape is None:
        return value, False
    s = str(value).strip()
    original = s

    # Strip a leading "the answer is …" style prefix (case-insensitive)
    low = s.lower()
    for p in _ANSWER_STRIP_PREFIXES:
        if low.startswith(p):
            s = s[len(p):].lstrip(" :\t")
            low = s.lower()
            break

    # Drop surrounding quotes
    s = s.strip().strip('"\'')

    if answer_shape == ANSWER_INTEGER:
        # Drop a trailing sentence-final period.
        s = s.rstrip(".")
        # Drop trailing unit tokens: "5 files" → "5", "5 unique IPs" → "5"
        tokens = s.split()
        while len(tokens) > 1 and tokens[-1].lower().strip(".,") in _ANSWER_UNIT_TOKENS:
            tokens.pop()
        s = " ".join(tokens).strip()
        # If still multi-token, try to find the last pure-int token.
        if not re.fullmatch(r"-?\d+", s):
            nums = re.findall(r"-?\d+", s)
            if nums:
                s = nums[-1]
    elif answer_shape == ANSWER_SIZE:
        # Collapse whitespace inside "42.5 MB" → "42.5MB"
        s = re.sub(r"(\d)\s+([KMGT]?B\b)", r"\1\2", s, flags=re.IGNORECASE)
        # Strip trailing ".0" from decimal sizes: "50.0K" → "50K".
        # The evaluator (size-match.py) calls int() on the numeric part, which
        # raises ValueError on "50.0", causing the comparison to fail silently.
        s = re.sub(r"(\d+)\.0+([KMGTP]?B?)\b", r"\1\2", s, flags=re.IGNORECASE)
        s = s.rstrip(".")
    elif answer_shape == ANSWER_STRING:
        s = s.rstrip(".")
        # Strip trailing newline already handled by .strip() above.
    elif answer_shape == ANSWER_PATH:
        s = s.rstrip("/").rstrip(".")

    return s, (s != original)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime — glues H0..H5 together
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OSHarnessRuntime:
    config: OSHarnessConfig
    task_ctx: Optional[OSTaskContext] = field(default=None)
    state: OSShellState = field(default_factory=OSShellState)
    force_next_action: Optional[Dict[str, Any]] = field(default=None)
    _last_hint: Optional[str] = field(default=None)
    _h5_injected_cold: bool = field(default=False)

    # ── H0 ──────────────────────────────────────────────────────────────────

    def init_task(self, description: str) -> None:
        self.task_ctx = parse_task_context(description)
        self.state = OSShellState()
        self.state.h4_fired_last_round = False
        self.force_next_action = None
        self._last_hint = None
        self._h5_injected_cold = False

    # ── H5 cold-start ───────────────────────────────────────────────────────

    def cold_start_skill_hints(self) -> List[Dict[str, str]]:
        if not self.config.h5_enabled or self.task_ctx is None:
            return []
        skills = retrieve_os_skills(
            task_type=self.task_ctx.task_type,
            query=self.task_ctx.raw_description,
            top_k=self.config.h5_top_k,
        )
        result: List[Dict[str, str]] = []
        for skill in skills:
            text = _truncate_to_word_budget(
                skill["text"], self.config.h5_cold_start_max_words
            )
            result.append({
                "id": skill["id"],
                "text": text,
                "trigger": "cold_start",
                "token_cost": str(len(text.split())),
            })
        self._h5_injected_cold = True
        return result

    # ── H2 ──────────────────────────────────────────────────────────────────

    def pre_validate_action(
        self,
        tool_name: Optional[str],
        raw_value: Optional[str],
    ) -> Dict[str, Any]:
        """Validate a (tool_name, raw_value) pair before the env executes it.

        Safety filter only — rescue parsing happens upstream in task.py because
        it needs access to the raw assistant content, not just the tool call.
        Returns: {action_name, action_value, blocked, reason}.
        """
        response: Dict[str, Any] = {
            "action_name": tool_name,
            "action_value": raw_value,
            "blocked": False,
            "reason": "",
        }
        if not self.config.h2_enabled:
            return response

        # Force action consumption — task.py writes force_next_action ahead.
        if self.force_next_action:
            fa = self.force_next_action
            self.force_next_action = None
            response["action_name"] = fa["name"]
            response["action_value"] = fa.get("arguments", {}).get(
                "answer", fa.get("arguments", {}).get("script", fa.get("arguments", {}).get("thought", ""))
            )
            response["reason"] = "force_next_action"
            return response

        if tool_name == "bash_action" and raw_value:
            dpat = _is_dangerous_bash(raw_value)
            if dpat:
                response["blocked"] = True
                response["reason"] = f"dangerous_bash:{dpat}"
                return response
            # Duplicate-bash gate: if the last two bash commands match raw_value
            # AND we have a PLAUSIBLE candidate answer ready → promote H4
            # budget force to submit the answer.  Never auto-submit an
            # implausible candidate (0 on a
            # filter task, awk overflow, …) — better to let the agent retry
            # with a corrected pipeline.
            hist = self.state.bash_history
            if (
                len(hist) >= self.config.h2_repeat_bash_block_after
                and all(h == raw_value for h in hist[-self.config.h2_repeat_bash_block_after:])
                and self.state.candidate_numeric_answer
                and not self.state.candidate_implausible
                and self.task_ctx is not None
                and self.task_ctx.answer_shape == ANSWER_INTEGER
                and not self.state.answered
            ):
                self.force_next_action = {
                    "name": "answer_action",
                    "arguments": {"answer": self.state.candidate_numeric_answer},
                }
                response["blocked"] = True
                response["reason"] = "duplicate_bash_force_answer"
                return response

        return response

    # ── H1 ──────────────────────────────────────────────────────────────────

    def update_state_after_bash(self, script: str, output: str) -> None:
        if script:
            self.state.bash_history.append(script)
        self.state.last_output_raw = output or ""
        self.state.last_output_truncated = bool(output and _TRUNC_MARK in output)
        self.state.last_output_was_empty = (output or "").strip() == ""
        self.state.last_output_had_error = bool(output and _ERROR_RE.search(output))
        self.state.last_output_numeric_candidates = extract_numeric_candidates(output or "")
        if self.state.last_output_was_empty:
            self.state.empty_output_streak += 1
        else:
            self.state.empty_output_streak = 0
        if self.state.last_output_numeric_candidates:
            self.state.candidate_numeric_answer = self.state.last_output_numeric_candidates[-1]
        # Single-line string candidate (for largest/smallest tasks with filename answer)
        lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
        if len(lines) == 1 and not self.state.last_output_had_error:
            self.state.candidate_string_answer = lines[0]
        # Candidate health: mark implausible only for awk overflow / negative /
        # huge values.  The "zero on filter task" heuristic was removed after
        # v2 analysis showed 0 is the correct answer in several training samples
        # (e.g. "count files with 'error' in their name" when none exist).
        # Blocking a correct 0 caused task_limit; submitting a wrong 0 fails the
        # same way — so the heuristic had no net benefit.
        # Only apply the implausibility check when answer_shape is integer;
        # string/date candidates (e.g. "2023-10-02") may contain negative-looking
        # substrings ("-02") that would be falsely flagged otherwise.
        self.state.candidate_implausible = False
        cand = self.state.candidate_numeric_answer
        if cand is not None and (
            self.task_ctx is None or self.task_ctx.answer_shape == ANSWER_INTEGER
        ):
            if not is_plausible_numeric_candidate(cand):
                self.state.candidate_implausible = True

    def note_answer_submitted(self) -> None:
        self.state.answered = True

    def note_text_only_turn(self) -> None:
        self.state.text_only_streak += 1

    def reset_text_only_streak(self) -> None:
        self.state.text_only_streak = 0

    def note_rescue_hit(self) -> None:
        self.state.rescue_hits += 1

    # ── H4 (post-step monitor, incl. budget warn/force) ──────────────────────

    def post_step_monitor(self, remaining_rounds: int = 99) -> Dict[str, Any]:
        """Inspect H1 state and return a recovery prompt + optional force.

        Budget warn/force is a sub-branch inside this function so there is a
        single post-bash monitoring trigger rather than two separate insertion
        points.  remaining_rounds is round_limit - current_round_num (inclusive
        of this round that just finished).
        Updates st.h4_fired_last_round so the post-ls nudge (⑥) can fire next round.
        """
        response: Dict[str, Any] = {
            "audit_reason": "",
            "recovery_prompt": None,
            "force_action": None,
        }
        if not self.config.h4_enabled or self.task_ctx is None:
            self.state.h4_fired_last_round = False
            return response
        st = self.state
        ctx = self.task_ctx

        # Capture whether H4 fired in the PREVIOUS round (before we update the flag).
        prior_h4_fired = st.h4_fired_last_round

        def _return(r: Dict[str, Any]) -> Dict[str, Any]:
            st.h4_fired_last_round = bool(r.get("recovery_prompt"))
            return r

        # ⓪ Truncation — agent should refine, not re-run
        if st.last_output_truncated:
            response["audit_reason"] = "truncated"
            response["recovery_prompt"] = (
                "Harness: output was truncated. Refine the command — e.g. pipe to "
                "`wc -l` for a count, `head -20` for a sample, or narrow the filter. "
                "Do NOT re-run the same command."
            )
            return _return(response)

        # ① Error signatures
        if st.last_output_had_error:
            err_text = st.last_output_raw.lower()
            if "command not found" in err_text:
                response["audit_reason"] = "command_not_found"
                response["recovery_prompt"] = (
                    "Harness: command not found. Try an alternative "
                    "(ss instead of netstat, ip a instead of ifconfig) or check PATH."
                )
                return _return(response)
            if "no such file or directory" in err_text or "paths must precede expression" in err_text:
                response["audit_reason"] = "no_such_file"
                cands = extract_numeric_candidates(st.last_output_raw)
                extra = ""
                if cands:
                    extra = (
                        f" The '{cands[-1]}' in the output is from an error exit, "
                        f"NOT a real count — do NOT submit it."
                    )
                response["recovery_prompt"] = (
                    "Harness: target file or directory missing (path error or wrong glob)."
                    + extra
                    + " Fix the path: use `find ~ -name '*.EXT'` or `grep -r --include='*.EXT'`. "
                    "Never use `~/.*\\.EXT` glob — it doesn't expand correctly."
                )
                return _return(response)
            if "permission denied" in err_text:
                response["audit_reason"] = "permission_denied"
                response["recovery_prompt"] = (
                    "Harness: permission denied. Prefix with `sudo` if you are root in this "
                    "container, or inspect with `stat` / `ls -l` before changing the file."
                )
                return _return(response)

        # ② Empty output when a count was expected
        if (
            st.last_output_was_empty
            and ctx.task_type in (TASK_COUNT_FILES, TASK_COUNT_LINES, TASK_COUNT_MATCHES, TASK_COUNT_UNIQUE)
            and st.empty_output_streak >= 1
        ):
            response["audit_reason"] = "empty_count"
            response["recovery_prompt"] = (
                "Harness: empty output. The filter may be too strict — drop the "
                "extension/time/size constraint one step at a time or widen the directory."
            )
            return _return(response)

        # ③ Same bash in a row on unchanged output
        if (
            len(st.bash_history) >= self.config.h4_stall_window
            and len(set(st.bash_history[-self.config.h4_stall_window:])) == 1
        ):
            response["audit_reason"] = "bash_loop"
            if (
                st.candidate_numeric_answer
                and ctx.answer_shape == ANSWER_INTEGER
                and not st.candidate_implausible
            ):
                response["recovery_prompt"] = (
                    f"Harness: you have run the same command {self.config.h4_stall_window} times. "
                    f"The output already contains the answer ({st.candidate_numeric_answer}). "
                    f"Stop repeating — submit your final answer now."
                )
            else:
                st.bash_loop_count += 1
                alt = self._first_turn_hint(ctx)
                # After 2+ loop detections with no candidate, force finish to break deadlock
                if st.bash_loop_count >= 2:
                    response["force_action"] = {
                        "name": "finish_action",
                        "arguments": {"thought": "Harness forced finish after repeated loop with no answer found."},
                    }
                    response["recovery_prompt"] = (
                        "Harness: repeated loop detected with no answer. Terminating episode."
                    )
                elif alt:
                    response["recovery_prompt"] = (
                        "Harness: you are repeating the same command with no progress. "
                        "Try this instead — " + alt.removeprefix("Hint: ")
                    )
                else:
                    response["recovery_prompt"] = (
                        "Harness: you are repeating the same command. Try a different approach "
                        "or submit the best answer you have."
                    )
            return _return(response)

        # ④ Text-only streak (≥ threshold) — hard nudge to invoke a tool
        if st.text_only_streak >= self.config.h2_text_only_streak_force:
            response["audit_reason"] = "text_only_streak"
            response["recovery_prompt"] = (
                "Harness: no tool call was detected. "
                "You must call a tool — either run a bash command or submit your final answer."
            )
            return _return(response)

        # ⑤ grep -c per-file output detection — when bash output looks like
        # "file1.log:3\nfile2.log:1" (multiple file:count lines from grep -c),
        # the extracted candidate is the last count (wrong).  Warn and suggest
        # the correct summation approach.
        _output_lines = [ln.strip() for ln in (st.last_output_raw or "").splitlines() if ln.strip()]
        _grep_c_lines = [ln for ln in _output_lines if _GREP_C_LINE_RE.match(ln)]
        if len(_grep_c_lines) >= 2:
            response["audit_reason"] = "grep_c_output"
            response["recovery_prompt"] = (
                "Harness: this looks like grep -c per-file output (each line is 'file:count'). "
                "To get the TOTAL count, use: `grep PATTERN DIR | wc -l`, or sum these numbers with "
                "`awk -F: '{sum+=$NF} END{print sum}'`."
            )
            return _return(response)

        # ⑥ Post-ls nudge — H4 fired last round (path error / empty), agent ran
        # `ls` this round, and the listing shows files.  This catches the pattern
        # where the agent confirms file existence but then submits 0 without
        # actually running a count/extraction command.
        if (
            prior_h4_fired
            and st.bash_history
            and re.match(r"^\s*ls\b", st.bash_history[-1])
            and not st.last_output_was_empty
            and not st.last_output_had_error
            and not st.answered
        ):
            response["audit_reason"] = "post_ls_nudge"
            response["recovery_prompt"] = (
                "Harness: files confirmed. Now run your actual count/extraction command "
                "on these files using their full path. "
                "Do NOT submit 0 — you have not counted yet."
            )
            return _return(response)

        # ⑦ Budget warn/force — H4 sub-branch, fires after bash; net effect
        # is identical to a round-top check since both inject before next LLM call.
        if not st.answered:
            rem = remaining_rounds - 1  # rounds left after this one completes
            if (
                rem <= self.config.h4_budget_force_threshold
                and ctx.answer_shape == ANSWER_INTEGER
                and st.candidate_numeric_answer
                and not st.candidate_implausible
            ):
                response["audit_reason"] = "budget_force"
                response["force_action"] = {
                    "name": "answer_action",
                    "arguments": {"answer": st.candidate_numeric_answer},
                }
                response["recovery_prompt"] = (
                    f"[{rem} rounds left] Harness: forcing answer_action with "
                    f"'{st.candidate_numeric_answer}'."
                )
                return _return(response)
            if rem <= self.config.h4_budget_warn_threshold:
                response["audit_reason"] = "budget_warn"
                if (
                    st.candidate_numeric_answer
                    and ctx.answer_shape == ANSWER_INTEGER
                    and not st.candidate_implausible
                ):
                    response["recovery_prompt"] = (
                        f"[{rem} rounds left] You have a candidate answer "
                        f"({st.candidate_numeric_answer}). Submit it now."
                    )
                elif st.candidate_implausible:
                    response["recovery_prompt"] = (
                        f"[{rem} rounds left] Your last pipeline produced an "
                        "implausible value. Broaden the filter or switch tools before submitting."
                    )
                else:
                    response["recovery_prompt"] = (
                        f"[{rem} rounds left] If you have a candidate answer, "
                        "submit it immediately."
                    )

        return _return(response)


    # ── H4-E: state-driven per-step guidance ────────────────────────────────

    def step_guidance(
        self,
        round_num: int,
        max_rounds: int,
        h4_audit_active: bool = False,
    ) -> Optional[str]:
        """H4-E: Per-step guidance driven by H1 runtime state (candidate answers, bash history).

        h4_audit_active: True when H4 just fired a recovery_prompt this same
        round.  In that case, suppress the "submit" hint (②, string promotion)
        so the model only sees H4's recovery message and doesn't follow a
        conflicting "submit '0'" hint instead.  Lint (①) is still allowed.
        """
        if not self.config.h4_enabled or self.task_ctx is None:
            return None
        ctx = self.task_ctx
        st = self.state

        hint: Optional[str] = None

        # ① Semantic-consistency lint — only fires when the last bash looks like
        # it diverges from the parsed task intent AND the agent has no candidate
        # yet.  Suppressed when a candidate already exists: the lint would push
        # the agent away from a value it already found, which is net-harmful
        # (cf. idx 24, 809 where lint caused the agent to switch from the correct
        # answer to a wrong one).
        if (
            st.bash_history
            and not st.answered
            and st.candidate_numeric_answer is None
            and st.candidate_string_answer is None
        ):
            gaps = bash_semantic_gaps(ctx, st.bash_history[-1])
            if gaps:
                gap = gaps[0]
                hint = f"Harness lint: {gap}."

        # ② Promote candidate numeric answer — suppressed when H4 has an active
        # recovery prompt (h4_audit_active=True) so the two signals don't
        # conflict.  Also suppressed for implausible values (awk overflow, etc.).
        #
        # Round-0 special case (v5): on the FIRST bash call (round_num==0,
        # bash_history has exactly 1 entry), emit a verification nudge instead of
        # a direct "submit" hint.  Analysis of v4 failures showed 22/35 were
        # 1-shot wrong-command submissions: the agent ran a semantically wrong bash
        # (wrong pattern, wrong path, missing -i), got a number, and immediately
        # submitted because H5 said "submit 'N'".  The nudge asks the agent to
        # re-read the task before committing to the answer.
        #
        # From round 1+, the original submit hint fires as before.
        #
        # candidate==0 always gets a verification nudge (regardless of round).
        if not h4_audit_active:
            if hint is None and (
                ctx.answer_shape == ANSWER_INTEGER
                and st.candidate_numeric_answer
                and not st.candidate_implausible
                and not st.answered
            ):
                if st.candidate_numeric_answer == '0':
                    hint = (
                        "Harness: result is 0. Before submitting, verify your path "
                        "and filter are correct — run `ls` on the target directory to "
                        "confirm it exists and has the expected files. "
                        "If the answer truly is 0, submit it."
                    )
                elif round_num == 0 and len(st.bash_history) == 1:
                    # First bash — push for verification rather than immediate submission.
                    hint = (
                        f"Hint: your first command returned '{st.candidate_numeric_answer}'. "
                        f"Verify before committing: right directory? correct filter pattern? "
                        f"case-sensitive match? counting lines not files? "
                        f"Run a second command to confirm if unsure."
                    )
                else:
                    hint = (
                        f"Hint: the last output contains the likely answer "
                        f"'{st.candidate_numeric_answer}'. If that matches the task, submit it."
                    )
            # Suspicious-zero nudge when the candidate IS implausible
            elif hint is None and (
                ctx.answer_shape == ANSWER_INTEGER
                and st.candidate_implausible
                and st.candidate_numeric_answer in ("0", "-9223372036854775808")
                and not st.answered
            ):
                hint = (
                    "Harness: the pipeline produced 0 / an overflow sentinel. Your "
                    "filter is likely too strict — drop the extension or time "
                    "constraint, or inspect the data with `head -3`."
                )
            # Promote single-line string candidate for any string-answer task
            # (previously limited to LARGEST/SMALLEST; extended in v4 to catch
            # tasks like "find the date with most events" classified as TASK_OTHER).
            elif hint is None and (
                ctx.answer_shape in (ANSWER_STRING, ANSWER_PATH, None)
                and st.candidate_string_answer
                and not st.answered
                and not st.last_output_had_error
            ):
                hint = (
                    f"Hint: if '{st.candidate_string_answer}' is the answer, submit it."
                )

        # First-turn suggestion when no bash has been run yet (always allowed)
        if hint is None and round_num == 0 and not st.bash_history:
            hint = self._first_turn_hint(ctx)

        if hint is None:
            return None

        words = hint.split()
        if len(words) > self.config.h4_hint_max_words:
            hint = " ".join(words[: self.config.h4_hint_max_words])

        if hint == self._last_hint:
            return None
        self._last_hint = hint
        return hint

    def _first_turn_hint(self, ctx: OSTaskContext) -> Optional[str]:
        path = ctx.target_path or "~"
        ext = ctx.extension_filter
        if ctx.task_type == TASK_COUNT_FILES:
            bits = [f"find {path} -type f"]
            if ext:
                bits.append(f"-name '*.{ext}'")
            if ctx.time_filter_days is not None:
                bits.append(f"-mtime -{ctx.time_filter_days}")
            if ctx.size_filter_bytes_gt is not None:
                bits.append(f"-size +{max(1, ctx.size_filter_bytes_gt // 1024)}k")
            if ctx.size_filter_bytes_lt is not None:
                bits.append(f"-size -{max(1, ctx.size_filter_bytes_lt // 1024)}k")
            cmd = " ".join(bits) + " | wc -l"
            return f"Hint: try `{cmd}`."
        if ctx.task_type == TASK_COUNT_MATCHES:
            if ext:
                return f"Hint: try `grep -{'i' if ctx.case_sensitive is False else ''}c PATTERN {path}/*.{ext}` per-file or `grep -{'i' if ctx.case_sensitive is False else ''}rh PATTERN {path} | wc -l` for totals."
            return f"Hint: use `grep -c PATTERN FILE` for per-file counts, or `grep -rh PATTERN {path} | wc -l` for totals."
        if ctx.task_type == TASK_COUNT_UNIQUE:
            return f"Hint: extract values with grep/awk then `sort -u | wc -l`. Start by inspecting `{path}` to see the data format."
        if ctx.task_type == TASK_COUNT_LINES:
            return f"Hint: use `wc -l` on the target file, or `find {path} -type f -exec cat {{}} + | wc -l` for a directory tree."
        if ctx.task_type == TASK_LARGEST:
            return f"Hint: try `find {path} -type f -printf '%s %p\\n' | sort -rn | head -1`."
        if ctx.task_type == TASK_SUM_SIZE:
            name_clause = f" -name '*.{ext}'" if ext else ""
            # Human-readable size needs bytes→format, because `du -h` outputs
            # strings like "24K" that awk cannot sum.  Keep the pipeline short.
            if ctx.answer_shape == ANSWER_SIZE:
                return (
                    f"Hint: `du -ch $(find {path} -type f{name_clause}) | tail -1 | awk '{{print $1}}'` "
                    f"— gives integer-valued output like '50K', not '50.0K'."
                )
            return (
                f"Hint: `find {path} -type f{name_clause} -printf '%s\\n' | "
                f"awk '{{s+=$1}} END{{print s}}'` for total bytes."
            )
        if ctx.task_type == TASK_AVERAGE:
            # Column-delimited files are the common case; generic but robust.
            return (
                f"Hint: run `head -3 {path}` first to see the delimiter, then "
                f"`awk -F: '$2~/^[0-9]+$/{{s+=$2;c++}} END{{if(c>0)printf \"%.0f\",s/c; else print 0}}' {path}`."
            )
        return None

    # ── H2 answer normalisation ──────────────────────────────────────────────

    def normalize_answer(self, value: str) -> Tuple[str, bool]:
        if self.task_ctx is None:
            return value, False
        return normalize_answer(value, self.task_ctx.answer_shape)
