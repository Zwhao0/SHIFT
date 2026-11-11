"""Supported complete Qwen MoE layouts; physical hosts always have eight GPUs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionLayout:
    world: int = 8

    def __post_init__(self):
        if type(self.world) is not int or self.world not in (4,8):
            raise ValueError('replica world must be four or eight ranks')

    @classmethod
    def from_config(cls, config):return cls(config.get('replica_world_size',8))

    @property
    def width(self):return self.world//2

    @property
    def tp(self):return self.world//4

    @property
    def affected(self):return self.stage_ranks(1)

    def stage_ranks(self, stage):
        if type(stage) is not int or stage not in (0, 1):
            raise ValueError('stage must be PP0 or PP1')
        return tuple(range(stage * self.width, (stage + 1) * self.width))

    def partial_scope(self, replica, ranks):
        if type(replica) is not int or replica not in (0, 1):
            raise ValueError('replica must be DP0 or DP1')
        ranks = tuple(ranks)
        if any(type(rank) is not int for rank in ranks) or ranks not in (self.stage_ranks(0), self.stage_ranks(1)):
            raise ValueError('partial migration requires one complete PP stage')
        return PartialScope(replica, ranks, self.width if ranks[0] == 0 else 0)

    def rank(self,role,device,full=False):
        if type(device) is not int or device not in range(self.world if full else self.width):
            raise ValueError('device outside actor scope')
        if full:return device
        if role not in ('A','B','C','D','E'):raise ValueError('unknown stage role')
        return device+(self.width if role in ('B','C','E') else 0)

    def local_data_rank(self,rank):return (rank%self.width)//self.tp


@dataclass(frozen=True)
class PartialScope:
    replica: int
    ranks: tuple
    coordinator_rank: int

    def owns(self, replica, rank):
        return replica == self.replica and rank in self.ranks

    def coordinates(self, replica, rank):
        return replica == self.replica and rank == self.coordinator_rank


def partial_roles(actor_role):
    source = actor_role if actor_role in ('B', 'C') else 'B'
    return source, 'C' if source == 'B' else 'B'
