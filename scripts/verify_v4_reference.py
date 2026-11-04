"""Audit the independent no-migration reference without trusting pass flags."""
import hashlib
import json
from pathlib import Path

from shift_full.v4_reference import build_signatures, compare_signature, validate_signatures


def verify(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    check(summary.get('reference_only') and summary.get('outer_manager'),'independent reference mode')
    check(summary.get('functional_pass') and not summary.get('error') and not summary.get('cleanup_errors'),
          'reference completed cleanly')
    steps=summary['plan']['cut']+summary['plan']['post_steps'];rows={}
    for role in 'ABDE':
        rows[role]=[]
        for device in range(4):
            events=[json.loads(line) for line in (root/f'{role}{device}.jsonl').read_text().splitlines() if line.strip()]
            rows[role].extend(events);label=role+str(device);replica=int(role in 'DE')
            rank=2*device+int(role in 'BE') if summary.get('cross_node_tp') else device+(4 if role in 'BE' else 0)
            check(bool(events) and all((x['role'],x['device'],x['rank'],x['replica'])==(role,device,rank,replica)
                  for x in events),'binding '+label)
            check(len({x['pid'] for x in events})==1,'PID '+label)
            commits=[x for x in events if x['kind']=='commit']
            check([x['step'] for x in commits]==list(range(1,steps+1)) and
                  all(x['phase']=='reference' and x['epoch']==0 and x['finite'] for x in commits),'commit coverage '+label)
            begin=[x for x in events if x['kind']=='outer_begin']
            outer=[x for x in events if x['kind']=='outer_commit']
            check([x['step'] for x in begin]==list(range(steps)) and
                  all(x['participants']==2 and x['participating_rank']==replica for x in begin),'quorum '+label)
            check([x['step'] for x in outer]==list(range(steps)) and
                  all(x['committed_step']==x['step']+1 and x['batches_committed']==2*(x['step']+1) for x in outer),
                  'Manager progress '+label)
            ready=[x for x in events if x['kind']=='outer_ready']
            check(len(ready)==1 and ready[0]['outer_epoch']==0 and
                  all(ready[0][k]==summary['outer_timeout'] for k in
                      ('timeout_seconds','quorum_timeout_seconds','connect_timeout_seconds')),'live Manager '+label)
            if summary['outer_audit']:
                audit=[x for x in events if x['kind']=='outer_gradient_audit']
                check([x['step'] for x in audit]==list(range(steps)) and
                      all(x['exact'] and x['buckets']>0 for x in audit),'independent AVG '+label)
            check(not any(x['kind'] in ('boundary_selected','freeze','outer_restored','activated','restored')
                          for x in events),'no migration or rewind '+label)
            done=[x for x in events if x['kind']=='done']
            check(len(done)==1 and done[0]['step']==steps and done[0]['identity_preserved'],'done '+label)
    try:
        built=build_signatures(rows,summary)
        saved=validate_signatures(json.loads((root/'reference_signatures.json').read_text()))
        check(built==saved,'saved signatures match independent events')
        for role,events in rows.items():
            for x in events:
                if x['kind']=='commit':
                    expected=saved['workers'][f'{x["replica"]}/{x["rank"]}'][str(x['step'])]
                    check(x['sample_ids']==expected['sample_ids'],'committed sample cursor')
    except (ValueError,KeyError) as error:errors.append(str(error))
    before=json.loads((root/'resources_before.json').read_text());after=json.loads((root/'resources_after.json').read_text())
    nodes={'node0','node1','node3'}
    clean=set(before)==set(after)==nodes and all(len(before[n]['company'])==8 and
        before[n]['company']==after[n]['company'] and not before[n]['extra_gpu'] and not after[n]['extra_gpu'] for n in nodes)
    check(clean,'resource/company PID audit')
    return dict(pass_all=not errors,errors=errors,resources_clean=clean,reference_only=True,
                logical_workers=16,reference_rank_steps=16*steps,independent_gradient_audit=summary['outer_audit'])


def verify_comparisons(root,summary,rows):
    """Recompute every cross-run comparison from signatures, ignoring exact flags."""
    root=Path(root);path=root/'reference_signatures.json';errors=[]
    try:
        reference=validate_signatures(json.loads(path.read_text()))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=summary['reference_sha256']:
            errors.append('independent reference manifest SHA mismatch')
        for role,devices in rows.items():
            for device,events in devices.items():
                rank=device+(4 if role in 'BCE' else 0);replica=int(role in 'DE')
                actual=[x for x in events if x['kind']=='reference_comparison']
                commits=[x for x in events if x['kind']=='commit']
                if [x['step'] for x in actual]!=[x['step'] for x in commits]:
                    errors.append('independent reference coverage '+role+str(device))
                if role!='C':
                    initial=[x for x in events if x['kind']=='reference_initial']
                    if len(initial)!=1 or initial[0].get('step')!=0:
                        errors.append('independent initial coverage '+role+str(device))
                    actual=initial+actual
                for x in actual:
                    if not compare_signature(reference,replica=replica,rank=rank,step=x['step'],
                        hashes=x['hashes'],samples=x['sample_ids'],manager=x['outer_manager']):
                        errors.append('independent signature mismatch '+role+str(device)+'/'+str(x['step']))
    except (ValueError,KeyError,FileNotFoundError) as error:errors.append('independent reference: '+str(error))
    return errors


def verify_retention(rows,cut):
    errors=[]
    for role,devices in rows.items():
        for device,events in devices.items():
            label=role+str(device);rank=device+(4 if role in 'BCE' else 0)
            ready=[x for x in events if x['kind']=='outer_ready']
            if any(x.get('service_epoch')!=0 for x in ready):errors.append('retained service epoch '+label)
            if role in 'ADE':
                boundary=[x for x in events if x['kind']=='outer_retention_boundary']
                retained=[x for x in events if x['kind']=='outer_retained']
                if len(boundary)!=1 or len(retained)!=1:
                    errors.append('retention evidence '+label);continue
                before,after=boundary[0],retained[0]
                identity=before.get('identity',{});training=before.get('training_identity',{})
                if (set(identity)!={'manager','server','client','executor','pg','service_epoch'} or
                    identity!=after.get('identity') or identity.get('service_epoch')!=0):
                    errors.append('Manager/server/client/PG identity '+label)
                if (set(training)!={'model','optimizer','ddp','parameters','adam','buckets'} or
                    not all(training.get(k) for k in ('parameters','adam','buckets')) or
                    training!=after.get('training_identity')):
                    errors.append('training/Adam/bucket storage identity '+label)
                expected=dict(step=cut,batches_committed=2*cut)
                if before.get('manager_state')!=expected or after.get('manager_state')!=expected:
                    errors.append('retained Manager progress '+label)
                if any(x['kind']=='outer_closed' and x['monotonic']<after['monotonic'] for x in events):
                    errors.append('retained Manager closed early '+label)
            for x in events:
                if x['kind']!='outer_begin':continue
                after=x['outer_epoch']==1
                namespace=1 if after and rank>=4 else 0
                configures=2 if after and role=='E' else 1
                if x.get('migration_namespace')!=namespace or x.get('pg_configures')!=configures:
                    errors.append('affected-only channel namespace/configuration '+label)
    return errors


def summarize(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());audit=verify(root)
    if not audit['pass_all']:raise ValueError(audit['errors'])
    return dict(name=summary['name'],reference_only=True,plan=summary['plan'],outer_manager=True,
        cross_node_tp=bool(summary.get('cross_node_tp')),
        outer_audit=summary['outer_audit'],outer_timeout=summary['outer_timeout'],functional_pass=True,
        resources_clean=audit['resources_clean'],cleanup_errors=summary['cleanup_errors'],
        source_sha256=summary['source_sha256'],independent_audit=audit,
        reference_sha256=hashlib.sha256((root/'reference_signatures.json').read_bytes()).hexdigest(),
        comparison_method='nine_state_sha256_not_full_tensor_dump')
