"""Build FLASH's epoch-local MONITOR input from committed SHIFT worker events."""


class EpochSamples:
    def __init__(self, snapshot):
        self.epoch = snapshot['epoch']
        self.first = self.next = snapshot['first_step']
        self.world = snapshot['replica_world_size']
        self.owners = {int(rank): worker['actor'] for rank, worker in snapshot['workers'].items()}
        self.offsets = dict.fromkeys(self.owners.values(), 0)
        self.pending = {}

    def collect(self, rows):
        for actor, offset in self.offsets.items():
            events = rows[actor]
            for event in events[offset:]:
                if (event['kind'] != 'session_commit' or event['session_epoch'] != self.epoch
                        or event['step'] < self.next or event['participants'] != 2
                        or not event['contributed'] or not event['finite']):
                    continue
                rank = event['replica'] * self.world + event['rank']
                if self.owners.get(rank) == actor:
                    self.pending.setdefault(event['step'], {})[rank] = event['seconds']
            self.offsets[actor] = len(events)

        while len(self.pending.get(self.next, ())) == len(self.owners):
            times = self.pending.pop(self.next)
            yield dict(iteration=self.next - self.first + 1, step=self.next,
                       time_ms=1000 * times[0], active_workers=len(times))
            self.next += 1
