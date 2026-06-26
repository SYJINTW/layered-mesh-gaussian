"""Post-hoc per-stage timing from a pipeline log's [dd/mm HH:MM:SS] timestamps.

Usage: python parse_timing.py <log_file> [<log_file> ...]

Heuristic: maps the first timestamp at/after each stage's start-marker to that stage,
then stage_secs = next_marker_time - this_marker_time. Approximate (depends on log
phrasing) but needs no instrumentation — captures runs that predate the JSON timers.
For exact numbers use the per-stage {train,render}_timing*.json written by the pipeline.
"""
import re, sys, datetime

TS = re.compile(r"\[(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})\]")
# stage -> substring that marks its start (first match wins)
MARKERS = [
    ("warmup",  "Warmup"),
    ("train",   "Start Training"),
    ("render",  "Rendering progress"),
    ("metrics", "Evaluating"),
]

def parse(path):
    rows = []  # (stage, datetime) at first marker hit
    seen = set()
    last_ts = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = TS.search(line)
            if m:
                d, mo, H, M, S = map(int, m.groups())  # log is dd/mm
                last_ts = datetime.datetime(2000, mo, d, H, M, S)
            for stage, sub in MARKERS:
                if stage not in seen and sub in line and last_ts is not None:
                    rows.append((stage, last_ts)); seen.add(stage)
    print(f"\n== {path} ==")
    if not rows:
        print("  no recognizable stage markers"); return
    for i, (stage, t) in enumerate(rows):
        end = rows[i + 1][1] if i + 1 < len(rows) else last_ts
        print(f"  {stage:8s} {int((end - t).total_seconds()):6d} s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        parse(p)
