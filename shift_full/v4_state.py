"""Explicit runtime readiness and optional development state auditing."""
from collections.abc import Mapping


def boundary_validation(config):
    mode=config.get('boundary_validation')
    if mode is None:mode='runtime' if config.get('performance_no_audit') else 'full'
    if mode not in ('full','runtime'):raise ValueError('invalid boundary validation mode')
    return mode


def boundary_descriptor(packet):
    return {k:packet[k] for k in ('schema','source_rank','step','v4_epoch','v4_owner','switch_round')}


def validate_candidate_state(packet, *, rank, cut, epoch, target, round_id, source_replica,
                             stage, optimizer, scheduler, rng, progress, fixed_replicas=True):
    """Check version and complete tensor structure before applying any state.

    No tensor values are copied to CPU or hashed here. Strict load and the
    completed CUDA restore establish readiness; byte equality is a debug audit.
    """
    import torch
    from .planned_state import validate_state
    from .v4_outer import validate_progress
    validate_state(packet,rank=rank,step=cut,audit=False)
    if (packet.get('v4_epoch')!=epoch or packet.get('v4_owner')!=target or
        packet.get('switch_round')!=round_id or packet.get('source_replica')!=source_replica):
        raise ValueError('stale candidate boundary or wrong source')
    manager=packet.get('outer_manager')
    if fixed_replicas:
        valid=validate_progress(manager)==dict(step=cut,batches_committed=2*cut)
    else:
        valid=(isinstance(manager,Mapping) and set(manager)=={'step','batches_committed'} and
               type(manager.get('step')) is int and manager['step']==cut and
               type(manager.get('batches_committed')) is int and cut<=manager['batches_committed']<=2*cut)
    if not valid:raise ValueError('candidate Manager progress mismatch')

    def structure(actual,expected,path):
        if isinstance(expected,torch.Tensor):
            if (not isinstance(actual,torch.Tensor) or actual.shape!=expected.shape or
                actual.dtype!=expected.dtype or actual.layout!=expected.layout or
                actual.device.type!=expected.device.type):
                raise ValueError('checkpoint tensor structure mismatch: '+path)
        elif isinstance(expected,Mapping):
            if not isinstance(actual,Mapping) or actual.keys()!=expected.keys():
                raise ValueError('checkpoint mapping mismatch: '+path)
            for k,v in expected.items():structure(actual[k],v,path+'/'+str(k))
        elif isinstance(expected,(tuple,list)):
            if type(actual) is not type(expected) or len(actual)!=len(expected):
                raise ValueError('checkpoint sequence mismatch: '+path)
            for i,v in enumerate(expected):structure(actual[i],v,path+'/'+str(i))
        elif type(actual) is not type(expected):
            raise ValueError('checkpoint scalar type mismatch: '+path)

    structure(packet['model'],stage.state_dict(),'model')
    structure(packet['all_buffers'],dict(stage.named_buffers()),'buffers')
    template=optimizer.state_dict()
    if not template.get('state') or not packet['optimizer'].get('state'):
        raise ValueError('candidate needs complete Adam state')
    structure(packet['optimizer'],template,'optimizer')
    structure(packet['scheduler'],scheduler.state_dict(),'scheduler')
    structure(packet['progress'],progress,'progress')
    for key,value in rng.items():structure(packet[key],value,key)
