"""Start the V4 controller in one bounded, server-owned cgroup after deploy."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from shift_full.control import Session, endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix',required=True)
    p.add_argument('--seq-len',type=int,default=512)
    p.add_argument('--cut',type=int,default=4)
    p.add_argument('--post-steps',type=int,default=10)
    p.add_argument('--no-oracle',action='store_true')
    p.add_argument('--gpu-prepare',action='store_true')
    p.add_argument('--gpu-warmup',action='store_true')
    p.add_argument('--slow-ms',type=float,default=0)
    p.add_argument('--reuse-local-groups',action='store_true')
    p.add_argument('--preconnect-state',action='store_true')
    p.add_argument('--defer-source-local-close',action='store_true')
    p.add_argument('--outer-manager',action='store_true')
    p.add_argument('--outer-audit',action='store_true')
    p.add_argument('--outer-timeout',type=float,default=120)
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--reference-run')
    p.add_argument('--source-stall-round',type=int,default=-1)
    p.add_argument('--source-deadline',type=float,default=2)
    p.add_argument('--cross-node-tp',action='store_true')
    p.add_argument('--transactional-switch',action='store_true',default=True)
    p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--candidate-request-step',type=int,default=0)
    p.add_argument('--between-steps',type=int,default=4)
    p.add_argument('--cancel-round',type=int,default=-1)
    p.add_argument('--stall-restore-rank',type=int,default=-1)
    p.add_argument('--restore-deadline',type=float,default=10)
    p.add_argument('--fault-round',type=int,default=0)
    p.add_argument('--overlap-peer', action='store_true')
    p.add_argument('--retain-outer-managers', action='store_true')
    p.add_argument('--fail-restore-rank',type=int,default=-1)
    p.add_argument('--trace-step',type=int,default=0)
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--post-layout', action='store_true', help='no-fault timing control for the final cross-host PP layout')
    p.add_argument('--unified', action='store_true', default=True, help='batch8 random data; complete initial DP replica per host')
    p.add_argument('--legacy-layout',dest='unified',action='store_false',help='explicit historical batch16 layout')
    p.add_argument('--performance-no-audit', action='store_true')
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--boundary-validation',choices=('full','runtime'),help='development hashes or runtime readiness; performance mode defaults to runtime')
    p.add_argument('--final-reference-run',help='independent final audit after the timed window')
    p.add_argument('--digest-engine',choices=('reference','staged'),default='reference')
    a=p.parse_args()
    from shift_full.v4_state import boundary_validation
    a.boundary_validation=boundary_validation(vars(a))
    if a.performance_no_audit:a.no_oracle=True
    if a.boundary_validation=='runtime' and not a.performance_no_audit:p.error('runtime readiness requires performance-no-audit')
    if a.fail_restore_rank!=-1 and a.boundary_validation!='full':p.error('silent state corruption injection requires full boundary audit')
    if a.final_reference_run and (not a.performance_no_audit or a.reference_run or a.reference_only):p.error('final reference requires migrating performance mode')
    if a.final_reference_run and not all(x.isalnum() or x in '_-' for x in a.final_reference_run):p.error('invalid final reference suffix')
    if a.post_layout and not (a.unified and a.reference_only and a.performance_no_audit):p.error('post-layout is a unified no-fault performance control only')
    if a.unified and a.cross_node_tp:p.error('unified TP must remain within a node')
    if a.performance_no_audit and (a.outer_audit or a.reference_run):p.error('performance run cannot consume audited reference')
    if a.transactional_switch:
        a.outer_manager=True;a.overlap_peer=True;a.retain_outer_managers=True
        if not a.reference_only:
            a.retain_outer_managers=True;a.gpu_prepare=True;a.gpu_warmup=True
            a.reuse_local_groups=True;a.preconnect_state=True;a.defer_source_local_close=True
    if a.cross_node_tp and not a.transactional_switch:p.error('cross-node TP requires transactional switching')
    if a.cross_node_tp:
        a.reuse_local_groups=False;a.defer_source_local_close=False;a.gpu_warmup=False
    if a.cycles not in range(1,4) or a.between_steps<1 or a.restore_deadline<=0 or a.source_deadline<=0:
        p.error('bounded cycles 1..3, positive between steps and restore deadline required')
    if a.candidate_request_step<0 or (a.candidate_request_step and (not a.transactional_switch or a.reference_only)):p.error('candidate request step requires migrating transactional mode')
    affected=[1,3,5,7] if a.cross_node_tp else [4,5,6,7]
    if a.transactional_switch and (a.source_stall_round not in range(-1,a.cycles) or a.cancel_round not in range(-1,a.cycles) or a.fault_round not in range(a.cycles) or
        (a.stall_restore_rank!=-1 and a.stall_restore_rank not in affected)):
        p.error('fault round/rank must belong to this bounded switching plan')
    if sum((a.source_stall_round!=-1,a.cancel_round!=-1,a.stall_restore_rank!=-1,a.fail_restore_rank!=-1))>1:
        p.error('use one injected fault per controlled run')
    if a.gpu_warmup and not a.gpu_prepare:p.error('GPU warmup requires isolated GPU preparation')
    if a.reuse_local_groups and not a.gpu_prepare:p.error('local group reuse requires GPU preparation')
    if a.defer_source_local_close and not a.reuse_local_groups:p.error('deferred source close requires local group preservation')
    if a.outer_audit and not a.outer_manager:p.error('outer gradient audit requires outer Manager')
    if a.fail_restore_rank!=-1 and (not a.outer_manager or (not a.no_oracle and not a.transactional_switch) or a.fail_restore_rank not in ([1,3,5,7] if a.cross_node_tp else [4,5,6,7])):
        p.error('restore failure requires outer manager, no oracle, and rank 4..7')
    if (a.reference_only or a.reference_run or a.retain_outer_managers) and not a.outer_manager:
        p.error('independent reference and Manager retention require outer Manager')
    if a.overlap_peer and (not a.outer_manager or (not a.reference_only and not a.retain_outer_managers)):
        p.error('peer overlap requires outer Manager retention (or reference-only)')
    if a.reference_only and (a.reference_run or a.fail_restore_rank != -1):
        p.error('reference-only cannot migrate, inject failure or consume a reference')
    if a.reference_run and (a.no_oracle or not all(x.isalnum() or x in '_-' for x in a.reference_run)):
        p.error('reference-run requires a valid suffix and oracle enabled')
    if a.outer_manager and not (a.no_oracle or a.reference_only or a.reference_run):
        p.error('outer oracle requires --reference-run; create it with --reference-only first')
    if not a.suffix or not all(x.isalnum() or x in '_-' for x in a.suffix):p.error('invalid suffix')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket=f'ssh-v4-start-{os.getpid()}.sock')
    root='/workspace/SHIFT';unit='shift-v4-'+a.suffix+'-controller'
    args=['--suffix',a.suffix,'--seq-len',str(a.seq_len),'--cut',str(a.cut),'--post-steps',str(a.post_steps)]
    args+=['--seed',str(a.seed),'--boundary-validation',a.boundary_validation,'--digest-engine',a.digest_engine]
    if a.final_reference_run:args+=['--final-reference-run',a.final_reference_run]
    if a.unified:args+=['--unified']
    else:args+=['--legacy-layout']
    if a.post_layout:args+=['--post-layout']
    if a.performance_no_audit:args+=['--performance-no-audit']
    if a.no_oracle:args+=['--no-oracle']
    if a.gpu_prepare:args+=['--gpu-prepare']
    if a.gpu_warmup:args+=['--gpu-warmup']
    if a.reuse_local_groups:args+=['--reuse-local-groups']
    if a.preconnect_state:args+=['--preconnect-state']
    if a.defer_source_local_close:args+=['--defer-source-local-close']
    if a.outer_manager:args+=['--outer-manager']
    if a.outer_audit:args+=['--outer-audit']
    args+=['--outer-timeout',str(a.outer_timeout),'--fail-restore-rank',str(a.fail_restore_rank)]
    if a.reference_only:args+=['--reference-only']
    if a.reference_run:args+=['--reference-run',a.reference_run]
    if a.retain_outer_managers:args+=['--retain-outer-managers']
    if a.overlap_peer:args+=['--overlap-peer']
    if a.transactional_switch:args+=['--transactional-switch']
    if a.deterministic:args+=['--deterministic']
    args+=['--trace-step',str(a.trace_step)]
    if a.cross_node_tp:args+=['--cross-node-tp']
    args+=['--source-stall-round',str(a.source_stall_round),'--source-deadline',str(a.source_deadline)]
    args+=['--candidate-request-step',str(a.candidate_request_step)]
    args+=['--cycles',str(a.cycles),'--between-steps',str(a.between_steps),'--cancel-round',str(a.cancel_round),
        '--stall-restore-rank',str(a.stall_restore_rank),'--restore-deadline',str(a.restore_deadline),'--fault-round',str(a.fault_round)]
    args+=['--slow-ms',str(a.slow_ms)]
    command=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec=1020',
        '--property=KillMode=control-group','--property=TimeoutStopSec=20','--working-directory='+root,
        '--property=StandardOutput=append:'+root+'/runtime/v3-reproduce/v4_'+a.suffix+'.controller.log',
        '--property=StandardError=inherit','--setenv=PYTHONPATH='+root,'--setenv=PYTHONDONTWRITEBYTECODE=1',
        '/usr/bin/python3','-m','scripts.v4_remote',*args]
    code='from pathlib import Path;import subprocess,json;assert Path('+repr(root+'/runtime/v1-controller/credentials')+').is_file(),"prepare required";subprocess.run('+repr(command)+',check=True);print(json.dumps(dict(unit='+repr(unit)+')))'
    try:
        s.ensure_connection(e)
        print(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
    finally:
        subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
