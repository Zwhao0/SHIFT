"""Admission rules for the first, deliberately single-replica V4 experiment."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    cut: int = 4
    post_steps: int = 10
    seq_len: int = 512
    microbatches: int = 4

    def __post_init__(self):
        if self.cut < 1 or self.post_steps < 1 or self.seq_len < 2 or self.microbatches < 1:
            raise ValueError('positive committed cut and experiment dimensions required')

    @staticmethod
    def logical_rank(role, device, cross_node_tp=False):
        if role not in ('A', 'B', 'C') or type(device) is not int or device not in range(4):
            raise ValueError('one of A/B/C and a device in 0..3 required')
        return 2*device + int(role!='A') if cross_node_tp else device + (0 if role == 'A' else 4)

    @staticmethod
    def authorize(role, epoch):
        if type(epoch) is not int or epoch not in (0, 1) or role not in (('A', 'B') if epoch == 0 else ('A', 'C')):
            raise ValueError('retired or unready logical owner')

    def admit(self, packet, rank):
        from .planned_state import validate_state
        validate_state(packet, rank=rank, step=self.cut, audit=True)
        if packet.get('v4_epoch') != 1 or packet.get('v4_owner') != 'C':
            raise ValueError('wrong replacement epoch/owner')
        if not packet['optimizer'].get('state'):
            raise ValueError('nonempty Adam state required')


class Ownership:
    """A closed epoch cannot be revived, including on the retained A process."""
    def __init__(self):
        self.epoch = 0

    def advance(self):
        if self.epoch != 0:
            raise ValueError('single replacement already activated')
        self.epoch = 1

    def authorize(self, role, epoch):
        if epoch != self.epoch:
            raise ValueError('stale training epoch')
        Plan.authorize(role, epoch)


def state_channel_matches(sent, received):
    if sent.get('transport') == 'packed_gpu' or received.get('transport') == 'packed_gpu':
        keys = ('transport', 'bytes', 'gpu_bytes', 'control_bytes', 'cpu_tensor_bytes',
                'hashes', 'oracle_hashes')
        return (all(k in sent and k in received and sent[k] == received[k] for k in keys)
                and sent['transport'] == 'packed_gpu' and sent['gpu_bytes'] > 0
                and 0 < sent['control_bytes'] <= 16*1024**2+8
                and 0 <= sent['cpu_tensor_bytes'] <= 4*1024**2
                and sent['bytes'] == sent['gpu_bytes'] + sent['control_bytes'])
    return sent['sha256'] == received['sha256'] and sent['bytes'] == received['bytes']
