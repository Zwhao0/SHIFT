"""Fetch the pinned Linux wheel manifest using mirrors and verify every archive."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse


def matches(path, wheel):
    if not path.exists() or path.stat().st_size != wheel['size']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == wheel['sha256']


def fetch(directory, wheel):
    name = wheel['filename']
    if Path(name).name != name or not name.endswith('.whl'):
        raise ValueError('invalid wheel filename in manifest')
    path = directory / name
    if matches(path, wheel):
        return
    source = urlparse(wheel['url'])
    if source.netloc == 'files.pythonhosted.org':
        mirrors = ['https://mirrors.aliyun.com/pypi' + source.path]
    elif source.netloc in ('download.pytorch.org', 'download-r2.pytorch.org'):
        mirrors = ['https://mirrors.aliyun.com/pytorch-wheels' + source.path.removeprefix('/whl'),
                   'https://mirrors.nju.edu.cn/pytorch' + source.path]
    else:
        raise ValueError('unrecognized wheel origin')
    tmp = path.with_suffix('.part')
    for url in [*mirrors, wheel['url']]:
        result = subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','1',
                                 '--connect-timeout','12','--max-time','360','--output',str(tmp),url],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode == 0 and matches(tmp, wheel):
            tmp.replace(path)
            print(name, flush=True)
            return
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f'could not download a verified {name}')


def main():
    root = Path(sys.argv[1])
    directory = root / 'runtime' / 'lab-wheels'
    directory.mkdir(exist_ok=True)
    manifest = root / 'runtime' / 'torch214-manifest' / 'wheels-linux-cp312.json'
    packages = json.loads(manifest.read_text())['packages']
    with ThreadPoolExecutor(max_workers=3) as executor:
        for _ in executor.map(lambda p: fetch(directory,p['wheels'][0]), packages):
            pass


if __name__ == '__main__':
    main()
