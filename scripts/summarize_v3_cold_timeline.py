"""Summarize same-worker cold-recovery timings from immutable collected logs."""
import argparse
import json
from pathlib import Path


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def summarize(status_path):
    status=json.loads(status_path.read_text())
    root=status_path.parent.parent/status['session']
    manifest=json.loads((root/'physical_manifest.json').read_text())
    result=dict(session=status['session'],sync_quorum=status.get('sync_quorum',False),roles={})
    for role in manifest:
        directory=root/('physical_'+role['physical_node'])/Path(role['directory']).name
        ranks={}
        for rank in range(8):
            es=rows(directory/f"events_g{role['generation']}_r{rank}.jsonl")
            recv=[e for e in es if e['kind']=='transport_recv']
            load=[e for e in es if e['kind']=='state_loaded']
            first=next((e for e in es if e['kind'] in ('commit','skipped')),None)
            ready=next(e for e in es if e['kind']=='ready')
            r=dict(sync_quorum=ready.get('sync_quorum'),first_attempt={k:first.get(k) for k in
                   ('kind','step','sample_ids','participants','contributed','global_batch','finite_gradients','schedule_seconds','gradient_check_seconds','decision_seconds')} if first else None)
            if recv and load:
                r.update(recv_seconds=recv[0]['seconds'],recv_to_loaded_seconds=load[0]['monotonic']-recv[0]['monotonic'],load_seconds=load[0]['load_seconds'])
            snapshots=[e for e in es if e['kind']=='snapshot']
            begins=[e for e in es if e['kind']=='snapshot_begin']
            if snapshots and begins:r['snapshot_seconds']=snapshots[0]['monotonic']-begins[0]['monotonic']
            ranks[str(rank)]=r
        result['roles'][f"{role['physical_node']}:rid{role['rid']}:g{role['generation']}"]=ranks
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('status',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.status)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),roles=list(result['roles']))))


if __name__=='__main__':main()
