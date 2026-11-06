"""Pinned-runtime preservation of drained, host-local V4 process groups.

Only TP2/PP2/EP2 with four devices per stage is supported. The default PG is
still rebuilt. Surviving subgroup backends are detached from the WORLD shutdown
list and registered in the new world using their new global rank mappings.
No installed Torch or Megatron source is modified.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys


SOURCE_HASHES = {
    'distributed_c10d.py': 'ffa3de81159c96c1a1042c8e2bce52fb604cd7e083675ebd7bed9a12d9c0143b',
    'parallel_state.py': '2edd2a8caea7746e160f16afbc1650a6a8e249085dfcf67997513673ac54593d',
    'megatron_adapter.py': 'c1da9ff1db8628290bdfdf372a43c5043d3f93ee04c7a6be3b4d94a5c4e3d76e',
}


def validate_runtime(c10d, state, adapter):
    for module in (c10d, state, adapter):
        path = Path(module.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASHES.get(path.name):
            raise RuntimeError('local group reuse requires the pinned source: '+path.name)


class LocalGroupCache:
    def __init__(self, dist, c10d, source_paths, stage_width=4):
        self.dist, self.c10d = dist, c10d
        if stage_width not in (2,4):raise ValueError('unsupported stage width')
        self.stage_width=stage_width
        self.sources = {str(Path(p).resolve()) for p in source_paths}
        self.saved = {}
        self.used = set()
        self.detached = False

    @staticmethod
    def local_members(ranks,stage_width=4):
        ranks = tuple(ranks)
        if not ranks or len({r//stage_width for r in ranks}) != 1:
            return None
        return tuple(r%stage_width for r in ranks)

    @contextmanager
    def creating(self, reuse=False):
        if reuse and not self.detached:
            raise RuntimeError('cannot reuse groups before drained detachment')
        original = self.dist.new_group
        def create(ranks, *args, **kwargs):
            frame = sys._getframe(1)
            path = str(Path(frame.f_code.co_filename).resolve())
            if path not in self.sources or args:
                raise RuntimeError('unsupported group constructor in pinned V4 adapter')
            members = tuple(ranks)
            local = self.local_members(members,self.stage_width)
            member = self.dist.get_rank() in members
            backend = kwargs.get('backend') or self.dist.get_backend()
            key = (path, frame.f_lineno, local, str(backend))
            if reuse and local is not None and member and key in self.saved:
                if key in self.used:
                    raise RuntimeError('duplicate local group reuse')
                group, old_store, backend_name, config = self.saved[key]
                # Advance the same global name sequence as ordinary new_group,
                # including nonmembers, so every rank agrees on new PG names.
                name = self.c10d._process_group_name(list(members), use_hashed_name=False)
                group._set_group_name(name)
                self.c10d._register_pg_in_world(group, backend_name, old_store, name, config,
                    rank_mapping={rank:i for i,rank in enumerate(members)}, add_to_default_tag=True)
                self.used.add(key)
                return group
            group = original(ranks, **kwargs)
            if not reuse and local is not None and member:
                if key in self.saved:
                    raise RuntimeError('ambiguous local group constructor')
                world = self.c10d._world
                backend_name, old_store = world.pg_map[group]
                self.saved[key] = (group, old_store, backend_name, world.pg_backend_config[group])
            return group
        self.dist.new_group = create
        try:
            yield
        finally:
            self.dist.new_group = original

    def detach(self):
        if self.detached or not self.saved:
            raise RuntimeError('local groups must be captured once before detaching')
        world = self.c10d._world
        if world.comms:
            raise RuntimeError('torchcomms is unsupported by the local reuse adapter')
        for group, *_ in self.saved.values():
            if group not in world.pg_names or group in world.pg_coalesce_state:
                raise RuntimeError('local group missing or still coalescing')
        # The pinned destroy_process_group iterates pg_names to shutdown native
        # backends, then clears all registries. Strong references keep these PGs
        # and their original stores alive while WORLD is destroyed.
        for group, *_ in self.saved.values():
            del world.pg_names[group]
        self.detached = True
        return len(self.saved)

    def finish(self, retain_catalog=False):
        if not self.detached:
            raise RuntimeError('groups were not detached before reuse')
        discarded = 0
        for key, (group, *_) in sorted(self.saved.items()):
            if key not in self.used:
                group.shutdown()
                discarded += 1
        counts = dict(reused=len(self.used), discarded=discarded)
        if retain_catalog:
            self.saved = {key:value for key,value in self.saved.items() if key in self.used}
        else:
            self.saved.clear()
        self.used.clear()
        self.detached = False
        return counts
