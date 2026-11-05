"""Run the approved v4 recovery tasks serially, auditing each before continuing."""
import argparse
import json
from pathlib import Path
from scripts.run_v41_suite import run
from scripts.v42_results import inspect


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=Path);args=parser.parse_args()
    for task in json.loads(args.manifest.read_text())['tasks']:
        run(task)
        result=inspect(task['name'],audit=True)
        audit=result['independent_audit']
        timing=audit.get('timing',{})
        print(json.dumps(dict(audited=task['name'],pass_all=audit['pass_all'],
            recovery_seconds=timing.get('request_to_first_common_commit_seconds'),
            windows={k:v['max_rank_step']['mean'] for k,v in timing.get('fixed_windows',{}).items()})),flush=True)


if __name__=='__main__':main()
