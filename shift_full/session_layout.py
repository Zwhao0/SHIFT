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
    def affected(self):return tuple(range(self.width,self.world))

    def rank(self,role,device,full=False):
        if type(device) is not int or device not in range(self.world if full else self.width):
            raise ValueError('device outside actor scope')
        if full:return device
        if role not in ('A','B','C','D','E'):raise ValueError('unknown stage role')
        return device+(self.width if role in ('B','C','E') else 0)

    def local_data_rank(self,rank):return (rank%self.width)//self.tp
