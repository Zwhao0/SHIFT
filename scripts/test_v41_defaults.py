"""The normal CLI must select the same batch8 profile as audited explicit flags."""
import argparse
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def parse_actual_main(filename, args):
    main=next(x for x in ast.parse(filename.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='main')
    statements=[]
    for statement in main.body:
        statements.append(statement)
        if isinstance(statement,ast.Assign) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute) and statement.value.func.attr=='parse_args':break
    namespace=dict(argparse=argparse,__doc__='CLI defaults')
    with patch.object(sys,'argv',[str(filename),'--suffix','test','--reference-only',*args]):
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),str(filename),'exec'),namespace)
    return namespace['a']


class DefaultsTest(unittest.TestCase):
    def test_default_matches_explicit_audited_profile(self):
        for name in ('start_v4.py','v4_remote.py'):
            filename=Path(__file__).with_name(name)
            default=parse_actual_main(filename,[])
            explicit=parse_actual_main(filename,['--unified','--transactional-switch'])
            self.assertTrue(default.unified and default.transactional_switch)
            self.assertFalse(default.cross_node_tp)
            self.assertEqual(vars(default),vars(explicit))

    def test_historical_batch16_requires_explicit_legacy_flag(self):
        for name in ('start_v4.py','v4_remote.py'):
            self.assertFalse(parse_actual_main(Path(__file__).with_name(name),['--legacy-layout']).unified)


if __name__=='__main__':unittest.main()
