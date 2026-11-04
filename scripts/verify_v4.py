"""Independent audit of collected V4 events, including all logical rank roles."""
import argparse
import json
from pathlib import Path
from shift_full.v4_protocol import state_channel_matches


def verify(root):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    if summary.get('outer_manager'):
        from scripts.verify_v4_outer import verify as verify_outer
        return verify_outer(root)
    plan=summary['plan'];cut=plan['cut'];post=plan['post_steps'];oracle=summary['oracle']
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    rows={r:{d:[json.loads(x) for x in (root/f'{r}{d}.jsonl').read_text().splitlines() if x.strip()]
             for d in range(4)} for r in 'ABC'}
    for role in 'ABC':
        for device,es in rows[role].items():
            rank=device+(0 if role=='A' else 4)
            check(all(e['role']==role and e['rank']==rank and e['device']==device for e in es),'role binding '+role+str(device))
            check(len({e['pid'] for e in es})==1,'PID continuity '+role+str(device))
            commits=[e for e in es if e['kind']=='commit']
            check(all(e['finite'] for e in commits),'finite '+role+str(device))
            for phase in ('before','oracle','after'):
                expected=(list(range(1,cut+1)) if phase=='before' and role!='C' else
                          list(range(cut+1,cut+post+1)) if
                          (phase=='oracle' and oracle and role!='C') or (phase=='after' and role!='B') else [])
                check([e['step'] for e in commits if e['phase']==phase]==expected,'commit coverage '+role+str(device)+'/'+phase)
            for e in commits:
                expected=[(e['step']-1)*plan['microbatches']*2+m*2+device//2 for m in range(plan['microbatches'])]
                check(e['sample_ids']==expected,'sample cursor '+role+str(device)+'/'+str(e['step']))
            if role!='B':
                if summary.get('reuse_local_groups'):
                    reused=[e for e in es if e['kind']=='local_groups_reused']
                    check(len(reused)==1 and reused[0]['identity_preserved'] and reused[0]['reused']>=4,
                          'retained local communication groups '+role+str(device))
                if summary.get('preconnect_state') and role=='C':
                    ready=[e for e in es if e['kind']=='state_channel_prepared']
                    received=[e for e in es if e['kind']=='state_received']
                    check(len(ready)==1 and len(received)==1 and ready[0]['monotonic']<received[0]['monotonic'],
                          'standby preconnected '+str(device))
                done=[e for e in es if e['kind']=='done']
                check(len(done)==1 and done[0]['step']==cut+post,'completion '+role+str(device))
                check(done and done[0]['identity_preserved'],'retained identity '+role+str(device))
                if oracle:
                    check(done and all(done[0]['comparison']['exact'].values()),'final nine states '+role+str(device))
                    check(done and all(x['unequal']==0 for x in done[0]['comparison']['tensors'].values()),'full tensor comparison '+role+str(device))
                    steps=[e for e in es if e['kind']=='oracle_step']
                    if 'gpu_prepare' in summary:
                        check(len(steps)==post and all(e['exact'] for e in steps),'nine states at every post step '+role+str(device))
            else:
                if summary.get('defer_source_local_close'):
                    observed=[e for e in es if e['kind']=='activation_observed']
                    closed=[e for e in es if e['kind']=='source_local_groups_closed']
                    sent=[e for e in es if e['kind']=='state_sent']
                    check(len(observed)==len(closed)==len(sent)==1 and
                          sent[0]['monotonic']<observed[0]['monotonic']<=closed[0]['monotonic'] and
                          closed[0]['reused']==0 and closed[0]['discarded']>=4,
                          'source local close after activation '+str(device))
                if summary.get('preconnect_state'):
                    ready=[e for e in es if e['kind']=='state_channel_prepared']
                    boundary=[e for e in es if e['kind']=='cut_boundary']
                    check(len(ready)==1 and ready[0]['monotonic']<boundary[0]['monotonic'],
                          'source channel ready before cut '+str(device))
                check(sum(e['kind']=='old_owner_rejected' for e in es)==1,'source fence '+str(device))
                check(sum(e['kind']=='retired' for e in es)==1,'source retired '+str(device))
            if role!='C' and 'requested_cut' in summary:
                check(sum(e['kind']=='old_epoch_rejected' for e in es)==1,'old epoch cannot revive '+role+str(device))
                boundary=[e for e in es if e['kind']=='boundary_selected']
                check(len(boundary)==1 and boundary[0]['step']==cut and boundary[0]['standby_ready'],
                      'ready-only boundary '+role+str(device))
    for device in range(4):
        source=next(e for e in rows['B'][device] if e['kind']=='freeze')
        target=next(e for e in rows['C'][device] if e['kind']=='restored')
        check(source['hashes']==target['hashes'],'source-target complete state '+str(device))
        a=rows['A'][device]
        frozen=next(e for e in a if e['kind']=='freeze')
        active=next(e for e in a if e['kind']=='activated')
        check(frozen['hashes']==active['hashes'],'healthy state across group rebuild '+str(device))
        check(frozen['identity_preserved'] and active['identity_preserved'],'healthy identities across switch '+str(device))
        sent=next(e for e in rows['B'][device] if e['kind']=='state_sent')
        received=next(e for e in rows['C'][device] if e['kind']=='state_received')
        check(state_channel_matches(sent,received),'channel integrity '+str(device))
        if summary.get('state_transport') == 'packed_gpu':
            check(sent['hashes']==source['hashes'],'packed snapshot signature '+str(device))
            for role,kind in (('B','transport_send'),('C','transport_recv')):
                es=rows[role][device]
                check(sum(x['kind']==kind for x in es)==1,'GPU transfer coverage '+role+str(device))
            check(sum(x['kind']=='snapshot_readonly_verified' for x in rows['B'][device])==1,
                  'packed source read-only '+str(device))
    before=json.loads((root/'resources_before.json').read_text())
    after=json.loads((root/'resources_after.json').read_text())
    resources_clean=all(len(before[n]['company'])==8 and before[n]['company']==after[n]['company']
                        and not after[n]['extra_gpu'] for n in before)
    check(resources_clean,'company PID preservation and no extra GPU processes')
    result=dict(pass_all=not errors,errors=errors,logical_ranks=8,healthy_processes_preserved=4,
                source_processes_fenced=4,post_steps=post,per_step_oracle=oracle and 'gpu_prepare' in summary,
                final_full_tensor_oracle=oracle,resources_clean=resources_clean)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result=verify(a.directory)
    (a.directory/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if not result['pass_all']:raise SystemExit(1)


if __name__=='__main__':main()
