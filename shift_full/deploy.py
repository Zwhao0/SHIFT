"""Deploy project source only; never archive credentials, results, or runtimes."""
import argparse
from pathlib import Path
import subprocess
import tarfile

from .control import Session, endpoints_from_env, CONTAINER_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-node3', action='store_true')
    args = parser.parse_args()
    session = Session.__new__(Session)
    session.root = Path(__file__).resolve().parents[1]
    session._sudo_password = None
    archive = session.root/'tmp/shift-code.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['shift_full','shift_min','configs','scripts','vendor/bin','runtime/torch214-manifest',
                     'README.md','RESULTS.md','AGENTS.md','handoff.md','CURRENT_PLAN.md','OPTIMIZATION_NOTES.md']:
            path = session.root/name
            if path.exists():
                tar.add(path, arcname=name, filter=lambda info: None if '__pycache__' in info.name else info)
    endpoints = endpoints_from_env()
    for name in ['node0','node1']+(['node3'] if args.include_node3 else []):
        endpoint = endpoints[name]
        session.ensure_connection(endpoint)
        remote = endpoint.host_root+'/shift-code.tar.gz'
        session.ssh(endpoint,['mkdir','-p',endpoint.host_root])
        subprocess.run(['scp','-q','-o',f'ControlPath={session.root/"tmp"/endpoint.socket}',
                        str(archive),f'test@{endpoint.ip}:{remote}'],check=True,timeout=60)
        session.ssh(endpoint,['tar','-xzf',remote,'-C',endpoint.host_root])
        session.docker(endpoint,['cp',remote,endpoint.container+':'+CONTAINER_ROOT+'/shift-code.tar.gz'])
        session.execute_endpoint(endpoint,['tar','-xzf',CONTAINER_ROOT+'/shift-code.tar.gz','-C',CONTAINER_ROOT])
        print(name,'deployed',flush=True)


if __name__ == '__main__':
    main()
