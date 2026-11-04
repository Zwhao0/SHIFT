"""Recompare immutable full snapshots in two bounded CPU shards, without training."""
import json
from scripts import v1_acceptance_remote as remote
from shift_full.control import endpoints_from_env


def compare_existing(root, source_suffix, suffix, out):
    status = json.loads((root/f'results/v3_b1_20261103_{source_suffix}/status.json').read_text())
    assert status['done'] and len(status['rows']) == 3
    cases = {row['mode']: row['session'] for row in status['rows']}
    manifest = json.loads((root/'results'/cases['flat']/'physical_manifest.json').read_text())
    node = next(row['physical_node'] for row in manifest if row['rid'] == 0 and row['generation'] == 0)
    e = endpoints_from_env()[node]
    probe = remote.ServerRotation.__new__(remote.ServerRotation)
    probe.root, probe._sudo_password = root, remote.PASSWORD
    units, commands, paths = [], [], []
    for i, ranks in enumerate(('0,1,2,3', '4,5,6,7')):
        unit = f'shift-v3-numcompare-{suffix}-{i}'
        path = root/'results'/cases['flat']/f'flat_tolerance_{suffix}_{i}.json'
        commands.append(['systemd-run', '--quiet', '--collect', '--wait', '--unit='+unit,
                         '--property=RuntimeMaxSec=1200', '--property=KillMode=control-group',
                         '--property=CPUQuota=200%', '--property=MemoryMax=32G',
                         '--working-directory='+str(root), '--setenv=PYTHONPATH='+str(root),
                         '--setenv=OMP_NUM_THREADS=2', str(root/'runtime/torch214/bin/python'),
                         '-m', 'shift_full.compare_reference',
                         str(root/'results'/cases['shift']/'rid0_g0'),
                         str(root/'results'/cases['flat']/'rid0_g0'),
                         '--steps', ','.join(map(str, status['steps'])), '--ranks', ranks,
                         '--threads', '2', '--output', str(path)])
        units.append(unit); paths.append(path)
    code = ('import subprocess,json; from concurrent.futures import ThreadPoolExecutor; '
            'commands='+repr(commands)+'; '
            'pool=ThreadPoolExecutor(max_workers=2); '
            'print(json.dumps(list(pool.map(lambda cmd: subprocess.run(cmd).returncode,commands))))')
    reports, exits = [], []
    try:
        exits = json.loads(probe.ssh(e, ['sudo','-S','-p','','python3','-c',code],
                                    sudo_input=True, timeout=1250))
        for i, path in enumerate(paths):
            code=('from pathlib import Path; p=Path('+repr(str(path))+'); '
                  'print((p if p.exists() else p.with_suffix(".partial.json")).read_text())')
            report = json.loads(probe.ssh(e, ['python3','-c',code], timeout=60))
            (out/f'flat_tolerance_part{i}.json').write_text(json.dumps(report,indent=2)+'\n')
            reports.append(report)
    finally:
        for i, unit in enumerate(units):
            try:
                journal=probe.ssh(e,['sudo','-S','-p','','journalctl','-u',unit,'--no-pager','-n','40'],sudo_input=True)
                (out/f'compare_part{i}_journal.txt').write_text(journal)
            except Exception as exc:
                (out/f'compare_part{i}_journal_error.txt').write_text(type(exc).__name__+'\n')
        cleanup = ('import subprocess; units='+repr(units)+'; '
                   '[subprocess.run(["systemctl","stop",u],capture_output=True,timeout=25) for u in units]; '
                   'assert all(subprocess.run(["systemctl","is-active",u],capture_output=True,text=True).stdout.strip() '
                   'not in ("active","activating","deactivating") for u in units)')
        probe.ssh(e, ['sudo','-S','-p','','python3','-c',cleanup], sudo_input=True)
    byrank = {str(step): {} for step in status['steps']}
    for report in reports:
        for step, ranks in report['byrank'].items():
            assert not (set(byrank[step]) & set(ranks))
            byrank[step].update(ranks)
    complete = all(set(ranks) == set(map(str,range(8))) for ranks in byrank.values())
    result = dict(source_suffix=source_suffix, complete=complete,
                  steps=status['steps'], ranks=list(range(8)), tolerances=reports[0]['tolerances'],
                  byrank=byrank, unit_exits=exits, training_started=False,
                  shards_complete=all(r['complete'] for r in reports))
    result['pass'] = complete and result['shards_complete'] and all(r['pass'] for r in reports)
    (out/'flat_tolerance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
