#!/usr/bin/env python3
"""Verify, safely unpack, and rebuild logical views of results/v1 evidence."""
import argparse, hashlib, json, shutil, tarfile
from pathlib import Path, PurePosixPath

NODES = ("node0", "node1", "node3")
META = ("controller.jsonl", "config.json", "physical_manifest.json", "fault_results.json")

def digest(path):
    h = hashlib.sha256(); size = 0
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(b); size += len(b)
    return h.hexdigest(), size

def safe_extract(archive, dest):
    root = dest.resolve(); dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            p = PurePosixPath(m.name)
            if p.is_absolute() or ".." in p.parts or not (m.isdir() or m.isfile()):
                raise ValueError("unsafe archive member: " + m.name)
            parts = [x for x in p.parts if x not in ("", ".")]
            if not parts:
                if m.isdir(): continue
                raise ValueError("empty archive file path")
            target = dest.joinpath(*parts)
            if root not in target.resolve().parents:
                raise ValueError("archive path escapes destination: " + m.name)
            if m.isdir(): target.mkdir(parents=True, exist_ok=True); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise ValueError("duplicate archive member: " + m.name)
            with tar.extractfile(m) as src, target.open("xb") as out: shutil.copyfileobj(src, out)

def mappings(manifest, physical):
    if isinstance(manifest, list):
        for x in manifest:
            rid, gen, node = int(x["rid"]), int(x["generation"]), x["physical_node"]
            if rid not in (0, 1) or "/" in node: raise ValueError("invalid rotation mapping")
            yield rid, physical / ("physical_" + node) / f"rid{rid}_g{gen}"
    elif isinstance(manifest, dict):
        for x in manifest.get("endpoints", []):
            node = x["physical_node"]
            rids = x.get("logical_rids") or ([int(node[-1])] if node in ("node0", "node1") else [])
            if "/" in node: raise ValueError("invalid endpoint mapping")
            for rid in rids:
                if int(rid) not in (0, 1): raise ValueError("invalid logical rid")
                yield int(rid), physical / ("physical_" + node)
    else: raise ValueError("unsupported physical_manifest schema")

def logical_views(manifest, physical, output):
    for rid, source in mappings(manifest, physical):
        if not source.is_dir(): raise ValueError("mapped physical generation missing: " + str(source))
        logical = output / f"node{rid}"; logical.mkdir(exist_ok=True)
        for path in source.rglob("*"):
            if path.is_file() and path.name.startswith(("events_g", "topology_g")):
                target = logical / path.name
                if target.exists():
                    if target.read_bytes() != path.read_bytes(): raise ValueError("logical view collision: " + path.name)
                else: shutil.copy2(path, target)

def unpack(evidence, output):
    evidence, output = evidence.resolve(), output.resolve()
    manifest = json.loads((evidence / "manifest.json").read_text())
    entries = {x["path"]: x for x in manifest.get("files", [])}
    raw = evidence / "raw"; meta_root = evidence / "sessions"
    sessions = sorted(p.name for p in raw.iterdir() if p.is_dir())
    output.mkdir(parents=True, exist_ok=True)
    if any((output / s).exists() for s in sessions): raise FileExistsError("output session already exists")
    for session in sessions:
        src_meta = meta_root / session; dest = output / session
        dest.mkdir()
        for name in META:
            src = src_meta / name
            if src.is_file(): shutil.copy2(src, dest / name)
            elif name != "fault_results.json": raise FileNotFoundError(src)
        phys = dest
        for node in NODES:
            rel = f"raw/{session}/{node}.tgz"; archive = evidence / rel
            entry = entries.get(rel)
            if not archive.exists() and entry is None: continue
            if not archive.is_file() or not entry: raise FileNotFoundError("unmanifested archive: " + rel)
            sha, size = digest(archive)
            if sha != entry["sha256"] or size != entry["size_bytes"]: raise ValueError("archive digest mismatch: " + rel)
            safe_extract(archive, phys / ("physical_" + node))
        logical_views(json.loads((dest / "physical_manifest.json").read_text()), phys, dest)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, default=Path("results/v1"))
    p.add_argument("--output", type=Path, default=Path("tmp/v1-evidence"))
    a = p.parse_args(); unpack(a.evidence, a.output)

if __name__ == "__main__": main()
