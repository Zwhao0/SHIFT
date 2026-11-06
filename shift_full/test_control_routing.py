"""CPU-only tests for physical endpoint routing and evidence collection."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from shift_full.control import (
    Endpoint,
    Session,
    endpoints_from_env,
    route_endpoint,
    runtime_command,
    server_password,
)


def new_session():
    session = Session.__new__(Session)
    session.root = Path('/tmp/shift-control-test')
    session.name = 'routing_test'
    session.config = {}
    session.seconds = 10
    session.endpoints = endpoints_from_env({})
    session.takeover_node3 = True
    session.gen = [0, 0]
    session._read_offsets = {}
    session._read_cache = {}
    session._logical_cache = [{}, {}]
    session._sudo_password = None
    session.started_endpoints = {}
    session._endpoint_generations = {}
    session.log = Mock()
    session.event = Mock()
    return session


def command_payload(code):
    line = next(line for line in code.splitlines() if line.startswith('cfg=json.loads('))
    return json.loads(ast.literal_eval(line[len('cfg=json.loads('):-1]))


class EndpointRoutingTests(unittest.TestCase):
    def test_two_node_route_is_unchanged_without_takeover(self):
        endpoints = endpoints_from_env({})
        self.assertEqual(route_endpoint(0, 1, endpoints).physical_node, 'node0')
        self.assertEqual(route_endpoint(1, 1, endpoints).physical_node, 'node1')

    def test_rid1_generation_one_moves_to_node3(self):
        endpoints = endpoints_from_env({})
        self.assertEqual(route_endpoint(1, 0, endpoints, True).physical_node, 'node1')
        self.assertEqual(route_endpoint(1, 1, endpoints, True).physical_node, 'node3')
        self.assertEqual(route_endpoint(1, 2, endpoints, True).physical_node, 'node3')
        self.assertEqual(route_endpoint(0, 1, endpoints, True).physical_node, 'node0')

    def test_node3_uses_isolated_nerdctl_namespace_and_sudo_stdin(self):
        endpoint = endpoints_from_env({})['node3']
        command = runtime_command(endpoint, ['exec', endpoint.container, 'true'])
        self.assertEqual(command[:6], ['sudo', '-S', '-p', '', '/usr/local/bin/nerdctl', '--namespace'])
        self.assertIn('shift', command)
        self.assertNotIn('password', ' '.join(command).lower())

    def test_password_reader_uses_non_whitespace_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'server.md'
            path.write_text('login\n机器密码：pass-with-s\nother\n', encoding='utf-8')
            self.assertEqual(server_password(path), 'pass-with-s')

    def test_host_login_override_does_not_replace_shared_sudo_password(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'server.md'
            path.write_text('192.0.2.13密码 login-three  机器密码/sudo/容器密码：shared-sudo\n')
            self.assertEqual(server_password(path), 'shared-sudo')
            self.assertEqual(server_password(path, '192.0.2.13'), 'login-three')
            self.assertEqual(server_password(path, '192.0.2.10'), 'shared-sudo')

    def test_node3_password_is_sent_via_stdin_only(self):
        session = new_session()
        with tempfile.TemporaryDirectory() as directory:
            session.root = Path(directory)
            (session.root / 'server.md').write_text('机器密码：sentinel-s\n', encoding='utf-8')
            endpoint = session.endpoints['node3']
            with patch('shift_full.control.sp.run', return_value=Mock(returncode=0, stdout='', stderr='')) as run:
                session.ssh(endpoint, ['sudo', '-S', '-p', ''], sudo_input=True)
            args = run.call_args.args[0]
            self.assertNotIn('sentinel-s', ' '.join(args))
            self.assertEqual(run.call_args.kwargs['input'], 'sentinel-s\n')

    def test_route_change_has_fresh_offsets_and_keeps_old_generation_without_duplicates(self):
        session = new_session()
        calls = []

        def execute(endpoint, cmd, timeout=45):
            payload = command_payload(cmd[-1])
            calls.append((endpoint.physical_node, payload))
            if endpoint.physical_node == 'node1':
                if payload['offsets'].get('supervisor.jsonl'):
                    rows, offsets = {}, payload['offsets']
                else:
                    rows = {
                        'events_g0_r0.jsonl': [{'kind': 'commit', 'step': 1}],
                        'supervisor.jsonl': [{'kind': 'injected', 'rid': 1, 'generation': 0}],
                    }
                    offsets = {'events_g0_r0.jsonl': 20, 'supervisor.jsonl': 30}
            else:
                rows = {
                    'events_g1_r0.jsonl': [{'kind': 'restored', 'step': 1}],
                    'supervisor.jsonl': [{'kind': 'restart_requested', 'rid': 1, 'generation': 1}],
                }
                offsets = {'events_g1_r0.jsonl': 22, 'supervisor.jsonl': 32}
            return json.dumps({'rows': rows, 'offsets': offsets})

        session.execute_endpoint = execute
        session.ensure_connection = Mock()
        first = session.read(1)
        session.read(1)  # Empty incremental packet must not duplicate cached rows.
        session.gen[1] = 1
        latest = session.read(1)

        self.assertEqual(len(first['events_g0_r0.jsonl']), 1)
        self.assertEqual(len(latest['events_g0_r0.jsonl']), 1)
        self.assertEqual(latest['events_g1_r0.jsonl'][0]['kind'], 'restored')
        node3_payload = next(payload for node, payload in calls if node == 'node3')
        self.assertEqual(node3_payload['offsets'], {})
        self.assertIn('injected', [row['kind'] for row in latest['supervisor.jsonl']])
        self.assertIn('restart_requested', [row['kind'] for row in latest['supervisor.jsonl']])

    def test_takeover_releases_start_and_retains_all_endpoints_for_cleanup(self):
        session = new_session()
        node0, node1, node3 = (session.endpoints[name] for name in ('node0', 'node1', 'node3'))
        session.started_endpoints = {'node0': node0, 'node1': node1}
        session._endpoint_generations = {'node0': {0}, 'node1': {0}}
        session.marked = []
        actions = []
        session.mark_endpoint = lambda endpoint, name, text='': (session.marked.append((endpoint.physical_node, name)), actions.append((name, endpoint.physical_node)))
        session.await_ = Mock(side_effect=lambda label, *args, **kwargs: (actions.append((label, None)) or [{}, {}]))
        session.ensure_connection = Mock()

        def start_supervisor(rid, endpoint, generation):
            actions.append(('launch', endpoint.physical_node))
            session.started_endpoints[endpoint.physical_node] = endpoint
            session._endpoint_generations.setdefault(endpoint.physical_node, set()).add(generation)

        session.start_supervisor = start_supervisor
        session.replace_rid1_with_node3(0, 'after_solo', 0)

        self.assertEqual(session.gen[1], 1)
        self.assertIn(('node1', 'STOP'), session.marked)
        self.assertIn(('node3', 'START'), session.marked)
        self.assertEqual({e.physical_node for e in session.physical_endpoints()},
                         {'node0', 'node1', 'node3'})
        self.assertLess(actions.index(('STOP', 'node1')), actions.index(('old_endpoint_cleanup_0', None)))
        self.assertLess(actions.index(('launch', 'node3')), actions.index(('START', 'node3')))

    def test_stop_request_covers_every_launched_physical_endpoint(self):
        session = new_session()
        endpoints = [session.endpoints[name] for name in ('node0', 'node1', 'node3')]
        session.started_endpoints = {ep.physical_node: ep for ep in endpoints}
        session._endpoint_generations = {'node0': {0}, 'node1': {0}, 'node3': {1}}
        session._read_cache = {ep.physical_node: {'supervisor.jsonl': []} for ep in endpoints}
        session.read_endpoint = Mock()
        session.mark_endpoint = Mock()

        session.request_stop_all(session.physical_endpoints(), False, 42)

        stopped = {call.args[0].physical_node for call in session.mark_endpoint.call_args_list
                   if call.args[1] == 'STOP'}
        self.assertEqual(stopped, {'node0', 'node1', 'node3'})


if __name__ == '__main__':
    unittest.main()
