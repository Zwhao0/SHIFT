"""Small remote host resource snapshot for one SHIFT session."""

import json


CODE = r'''
import csv
import io
import json
import os
import subprocess
import sys
import time


def number(value):
    value = value.strip()
    if not value or value.lower() in ("n/a", "[n/a]", "not supported"):
        return None
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


session = sys.argv[1] if len(sys.argv) > 1 else ""
result = {"session": session, "gpus": [], "gpu_error": None,
          "sample_time": time.time(), "cpu_clock_ticks_per_second": os.sysconf('SC_CLK_TCK')}
try:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=10, check=False,
    )
    if query.returncode:
        result["gpu_error"] = "nvidia-smi_exit_%d" % query.returncode
    else:
        for row in csv.reader(io.StringIO(query.stdout)):
            if len(row) != 4:
                continue
            result["gpus"].append({
                "index": number(row[0]), "memory_used_mib": number(row[1]),
                "utilization_gpu_percent": number(row[2]), "power_draw_w": number(row[3]),
            })
except subprocess.TimeoutExpired:
    result["gpu_error"] = "nvidia-smi_timeout_10s"
except OSError as error:
    result["gpu_error"] = "nvidia-smi_%s" % type(error).__name__

session_bytes = session.encode("utf-8", "ignore")
self_pid = os.getpid()
states = {}
process_count = 0
skipped_races = 0
missing_rss = 0
rss_kib_total = 0
utime_ticks_total = 0
stime_ticks_total = 0
for entry in os.listdir("/proc"):
    if not entry.isdigit() or int(entry) == self_pid:
        continue
    base = "/proc/" + entry
    try:
        with open(base + "/cmdline", "rb") as stream:
            cmdline = stream.read()
        if not session_bytes or session_bytes not in cmdline:
            continue
        with open(base + "/stat", "r") as stream:
            stat = stream.read()
        tail = stat.rsplit(")", 1)[1].split()
        state = tail[0]
        utime_ticks = int(tail[11])
        stime_ticks = int(tail[12])
        rss_kib = None
        with open(base + "/status", "r") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        rss_kib = int(fields[1])
                    break
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError,
            IndexError, ValueError):
        skipped_races += 1
        continue
    process_count += 1
    states[state] = states.get(state, 0) + 1
    utime_ticks_total += utime_ticks
    stime_ticks_total += stime_ticks
    if rss_kib is None:
        missing_rss += 1
    else:
        rss_kib_total += rss_kib

result["processes"] = {
    "count": process_count,
    "states": states,
    "vmrss_kib_sum": rss_kib_total,
    "vmrss_missing_count": missing_rss,
    "cpu_utime_ticks_sum": utime_ticks_total,
    "cpu_stime_ticks_sum": stime_ticks_total,
    "skipped_races_or_unreadable": skipped_races,
}
print(json.dumps(result, separators=(",", ":"), sort_keys=True))
'''


def snapshot(session, endpoint):
    """Run the stdlib probe in ``endpoint`` and return its parsed JSON object."""
    raw = session.execute_endpoint(endpoint, ["python", "-c", CODE, session.name], timeout=15)
    return json.loads(raw)
