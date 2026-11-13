"""Redis contract between the FLASH localizer and one SHIFT Lab session."""

import hashlib
import json
import re

from .session_control import Command
from .session_layout import SessionLayout


def connect(socket_path):
    from redis import Redis

    return Redis(unix_socket_path=socket_path, decode_responses=True,
                 socket_connect_timeout=0.5, socket_timeout=0.5)


def keys(session):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', session):
        raise ValueError('invalid FLASH session')
    prefix = f'shift:flash:v1:{session}'
    return f'{prefix}:membership', f'{prefix}:alerts'


def membership(session, epoch, first_step, layout, active, actors, command_id=None):
    workers = {}
    for replica in range(2):
        for rank in range(layout.world):
            actor, device = active[replica, rank]
            info = actors[actor]
            workers[str(replica * layout.world + rank)] = dict(
                replica=replica, rank=rank, actor=actor, generation=info['generation'],
                runner=info['node'], physical_node=info['physical_node'],
                pod_uid=info['pod_uid'], gpu_uuid=info['gpu_uuids'][device])
    return dict(schema=1, session=session, epoch=epoch, first_step=first_step,
                replica_world_size=layout.world, command_id=command_id, workers=workers)


def alert(snapshot, prediction, rank_map):
    """Convert a completed FLASH PROFILE prediction pinned to a membership snapshot."""
    rank = prediction['alarm_rank']
    if rank is None or prediction['manual_integration_only'] or prediction['abstention_reasons']:
        return None
    if type(rank) is not int or rank < 0:
        raise ValueError('invalid FLASH rank')
    if (not isinstance(rank_map, dict)
            or set(rank_map) != set(snapshot['workers'])
            or any(type(value) is not int for value in rank_map.values())
            or set(map(str, rank_map.values())) != set(snapshot['workers'])):
        raise ValueError('FLASH rank map must cover the membership exactly')
    logical_rank = rank_map[str(rank)]
    first, last = profile_window(prediction['iteration_ids'])
    if first < snapshot['first_step']:
        raise ValueError('PROFILE predates active membership')
    return dict(schema=1, session=snapshot['session'], epoch=snapshot['epoch'],
                rank=logical_rank, worker=snapshot['workers'][str(logical_rank)],
                first_step=first, last_step=last, kind='compute_slow')


def profile_window(steps):
    if (type(steps) is not list or len(steps) != 8
            or any(type(step) is not int or step < 1 for step in steps)
            or steps != list(range(steps[0], steps[0] + 8))):
        raise ValueError('FLASH PROFILE requires eight consecutive steps')
    return steps[0], steps[-1]


def command(snapshot, event, policy, progress, max_age_steps):
    """Fence stale or uncommitted observations before entering the normal command path."""
    if (type(event['schema']) is not int or event['schema'] != 1
            or event['kind'] != 'compute_slow' or event['session'] != snapshot['session']
            or type(event['epoch']) is not int or event['epoch'] != snapshot['epoch']):
        raise ValueError('FLASH alert does not match the active session and epoch')
    rank = event['rank']
    if type(rank) is not int or str(rank) not in snapshot['workers']:
        raise ValueError('FLASH alert has an invalid logical rank')
    worker = snapshot['workers'][str(rank)]
    if event['worker'] != worker:
        raise ValueError('FLASH alert does not match the active worker')
    first, last = event['first_step'], event['last_step']
    if (type(first) is not int or type(last) is not int or last - first != 7
            or first < snapshot['first_step'] or last > progress or progress - last > max_age_steps):
        raise ValueError('FLASH alert is outside the committed observation window')
    layout = SessionLayout(snapshot['replica_world_size'])
    if policy == 'v2' and worker['replica'] == 1:
        ranks = range(layout.world)
    elif policy == 'v4':
        ranks = layout.stage_ranks(worker['rank'] // layout.width)
    else:
        raise ValueError('FLASH alert is outside the supported migration scope')
    # The same profile cannot request twice after producer or transport retries.
    identity = [snapshot['session'], snapshot['epoch'], rank, first, last]
    identifier = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    request = dict(schema=1, session=snapshot['session'], id=f'flash-{identifier}',
                   expected_epoch=snapshot['epoch'], action='migrate', policy=policy,
                   source_replica=worker['replica'], target='auto', ranks=list(ranks),
                   replica_world_size=layout.world)
    Command.from_dict(request)
    return request


class Bridge:
    def __init__(self, client, session):
        self.client = client
        self.membership_key, self.alert_key = keys(session)
        self.cursor = '0-0'
        self.published = None

    def publish(self, snapshot):
        if self.published != snapshot:
            self.client.set(self.membership_key, json.dumps(snapshot))
            self.published = snapshot

    def read(self):
        rows = self.client.xread({self.alert_key: self.cursor}, count=1)
        if not rows:
            return None
        cursor, values = rows[0][1][0]
        return cursor, values.get('event')
