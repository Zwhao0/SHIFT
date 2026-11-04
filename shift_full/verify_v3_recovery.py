"""Schema-2 recovery gates in addition to the historical rotation validator."""
from pathlib import Path

from shift_full.verify_rotation import verify, _load_evidence

FIELDS={'model','optimizer','scheduler','progress','cpu_rng','cuda_rng','python_rng','buffers','tracker_rng'}


def modern_checks(mapping,roles,faults):
    errors=[]
    for role,events in roles.items():
        ready=[e for e in events if e.get('kind')=='ready']
        if len(ready)!=1 or ready[0].get('state_schema')!=2 or set(ready[0].get('hashes',{}))!=FIELDS:
            errors.append(f'{role}: complete schema2 ready audit required')
    current={0:0,1:0}
    for f in faults:
        target=f['target'];healthy=1-target;generation=f['generation'];step=f['restored_step'];joint=f['first_joint_step']
        current[target]=generation
        for rank in range(8):
            restored=[e for e in roles.get((target,generation,rank),[]) if e.get('kind')=='restored' and e.get('step')==step]
            if not restored or any(set(e.get('expected_hashes',{}))!=FIELDS or set(e.get('actual_hashes',{}))!=FIELDS
                                   or e.get('expected_hashes')!=e.get('actual_hashes') or not e.get('hashes_match') for e in restored):
                errors.append(f'{target}:{generation}:{rank}: schema2 restored state incomplete')
            sender=roles.get((healthy,current[healthy],rank),[])
            if not any(e.get('kind')=='snapshot_readonly_verified' and e.get('step')==step and e.get('tensors',0)>0 for e in sender):
                errors.append(f'{healthy}:{current[healthy]}:{rank}: source read-only send not verified')
            for rid in range(2):
                rows=roles.get((rid,current[rid],rank),[])
                for k in range(joint,joint+10):
                    matches=[e for e in rows if e.get('kind')=='commit' and e.get('step')==k]
                    if len(matches)!=1 or not all(e.get('participants')==2 and e.get('contributed') and e.get('finite_gradients') for e in matches):
                        errors.append(f'{rid}:{current[rid]}:{rank}: missing/invalid joint step{k}')
                first=next((e for e in rows if e.get('kind')=='commit' and e.get('step')==joint),{})
                if set(first.get('hashes',{}))!=FIELDS:errors.append(f'{rid}:{current[rid]}:{rank}: joint schema2 audit missing')
        # A pure-cold attempt must not have any prepared worker generation.
        if f.get('backup_mode')!='cold' or f.get('standby_resources'):
            errors.append('pure cold recovery cannot use prepared workers')
    return errors


def verify_v3(result_dir,expected_faults=1):
    root=Path(result_dir)
    report=verify(root,expected_faults,require_audit=True)
    errors=[]
    mapping,roles,_,_,faults=_load_evidence(root,errors)
    errors+=modern_checks(mapping,roles,faults)
    report['schema2_errors']=errors
    report['schema2_pass']=not errors
    report['pass']=report['pass'] and not errors
    return report
