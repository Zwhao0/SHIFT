"""Incremental JSONL reader helpers for Session.read()."""

import json
import os


def remote_code(directory, generation, offsets=None):
    """Return stdlib-only ``python -c`` source for reading one generation."""
    generation = int(generation)
    saved = {}
    for name, offset in (offsets or {}).items():
        try:
            value = int(offset)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            saved[os.path.basename(str(name))] = value
    payload = json.dumps(
        {"directory": os.fspath(directory), "generation": generation, "offsets": saved},
        separators=(",", ":"),
    )
    return f'''import glob,json,os
cfg=json.loads({payload!r})
base=cfg["directory"]
gen=cfg["generation"]
saved=cfg["offsets"]
paths=sorted(glob.glob(os.path.join(base,"events_g%d_r*.jsonl" % gen)))
supervisor=os.path.join(base,"supervisor.jsonl")
if os.path.isfile(supervisor): paths.append(supervisor)
rows={{}}
event_prefix="events_g%d_r" % gen
new_offsets={{name:value for name,value in saved.items()
              if name=="supervisor.jsonl" or (name.startswith(event_prefix) and name.endswith(".jsonl"))}}
for path in paths:
 name=os.path.basename(path)
 with open(path,"rb") as stream:
  stream.seek(0,os.SEEK_END)
  size=stream.tell()
  start=saved.get(name,0)
  if start>size: start=0
  stream.seek(start)
  chunk=stream.read()
 end=chunk.rfind(b"\\n")+1
 complete=chunk[:end]
 batch=[]
 for line in complete.splitlines():
  if not line.strip(): continue
  try: row=json.loads(line)
  except (ValueError,UnicodeError): continue
  if isinstance(row,dict): batch.append(row)
 rows[name]=batch
 new_offsets[name]=start+end
print(json.dumps({{"rows":rows,"offsets":new_offsets}},separators=(",",":")))
'''


def _time_key(row):
    try:
        return float(row.get("time", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def merge(cache, packet):
    """Merge a reader packet into ``{filename: sorted rows}`` bounded cache."""
    merged = {name: list(rows) for name, rows in (cache or {}).items()}
    incoming = (packet or {}).get("rows", {})
    for name, new_rows in incoming.items():
        by_kind = {}
        for row in merged.get(name, []) + list(new_rows or []):
            if not isinstance(row, dict) or row.get("kind") == "phase":
                continue
            by_kind.setdefault(row.get("kind"), []).append(row)
        kept = []
        for kind_rows in by_kind.values():
            kept.extend(kind_rows[-12:])
        merged[name] = sorted(kept, key=_time_key)
    return merged
