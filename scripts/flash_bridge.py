"""Pin a SHIFT membership or publish one automatic FLASH prediction."""

import argparse
import json
from pathlib import Path

from shift_full.flash_bridge import alert, connect, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--redis-socket', required=True)
    parser.add_argument('--session', required=True)
    subcommands = parser.add_subparsers(dest='action', required=True)
    pin = subcommands.add_parser('pin')
    pin.add_argument('--snapshot', type=Path, required=True)
    publish = subcommands.add_parser('publish')
    publish.add_argument('--snapshot', type=Path, required=True)
    publish.add_argument('--prediction', type=Path, required=True)
    publish.add_argument('--rank-map', type=Path, required=True)
    publish.add_argument('--observed-workers', type=Path, required=True,
                         help='collector-recorded Pod UID and GPU UUID by FLASH rank')
    args = parser.parse_args()

    client = connect(args.redis_socket)
    member_key, alert_key = keys(args.session)
    raw = client.get(member_key)
    if raw is None:
        parser.error('SHIFT has no published active membership')
    if args.action == 'pin':
        with args.snapshot.open('x') as stream:
            stream.write(raw + '\n')
        return

    snapshot = json.loads(args.snapshot.read_text())
    if snapshot != json.loads(raw):
        parser.error('membership changed during PROFILE')
    prediction = json.loads(args.prediction.read_text())
    if prediction.get('fault_labels_read') is not False:
        parser.error('prediction must come from inference without fault labels')
    rank_map = json.loads(args.rank_map.read_text())
    event = alert(snapshot, prediction, rank_map)
    if event is None:
        print(json.dumps(dict(status='no_alert')))
        return
    expected = {rank: {key: snapshot['workers'][str(logical_rank)][key]
                       for key in ('pod_uid', 'gpu_uuid')}
                for rank, logical_rank in rank_map.items()}
    if json.loads(args.observed_workers.read_text()) != expected:
        parser.error('PROFILE workers do not match the pinned membership')
    print(json.dumps(dict(status='published', stream_id=client.xadd(
        alert_key, {'event': json.dumps(event)}))))


if __name__ == '__main__':
    main()
