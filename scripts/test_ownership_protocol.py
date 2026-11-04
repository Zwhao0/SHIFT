"""Adversarial CPU checks of the P10 specification; no GPU safety claim."""
import threading
import unittest
from shift_full.ownership_protocol import Cut, LocalFence, Ownership


class OwnershipTest(unittest.TestCase):
    def setUp(self):
        self.cut = Cut(1, 6, 'fresh-run-nonce')
        self.p = Ownership(self.cut)

    def record(self, role, *proofs, ranks=range(8)):
        for rank in ranks:
            for proof in proofs:
                self.p.record(self.cut, role, rank, proof)

    def ready(self):
        self.record('B', 'fenced')
        self.record('A', 'drained')
        self.record('C', 'initial_audit', 'computed', 'gradient_audit')

    def test_no_outer_race_even_if_main_reaches_gate_before_control(self):
        f = LocalFence(self.cut)
        f.reserve(quorum_complete=True, state_readers_done=True)
        self.assertEqual(f.preouter(self.cut, local_work_drained=True), 'park')
        with self.assertRaises(RuntimeError): f.allow_old_outer()
        t = threading.Thread(target=lambda: f.control_fence(self.cut))
        t.start(); t.join(timeout=2)
        self.assertFalse(t.is_alive())
        f.close_old_manager()
        with self.assertRaises(RuntimeError): f.allow_old_outer()

    def test_control_ack_while_B_sleeping_cannot_close_or_abort_it(self):
        f = LocalFence(self.cut)
        f.reserve(quorum_complete=True, state_readers_done=True)
        acks = []
        t = threading.Thread(target=lambda: acks.append(f.control_fence(self.cut)))
        t.start(); t.join(timeout=2)
        self.assertEqual(acks, [self.cut])
        self.assertFalse(f.drained)
        with self.assertRaises(RuntimeError): f.close_old_manager()

    def test_incomplete_quorum_and_state_readers_reject_reservation(self):
        for q, r in [(False, True), (True, False), (False, False)]:
            with self.subTest(q=q, r=r), self.assertRaises(RuntimeError):
                LocalFence(self.cut).reserve(quorum_complete=q, state_readers_done=r)

    def test_stale_and_duplicate_fence(self):
        f = LocalFence(self.cut)
        with self.assertRaises(RuntimeError): f.control_fence(self.cut)
        f.reserve(quorum_complete=True, state_readers_done=True)
        for cut in [Cut(2, 6, self.cut.nonce), Cut(1, 7, self.cut.nonce), Cut(1, 6, 'old')]:
            with self.assertRaises(ValueError): f.control_fence(cut)
        self.assertEqual(f.control_fence(self.cut), f.control_fence(self.cut))

    def test_background_thread_cannot_report_training_drain(self):
        f = LocalFence(self.cut)
        f.reserve(quorum_complete=True, state_readers_done=True)
        errors = []
        def other():
            try: f.preouter(self.cut, local_work_drained=True)
            except RuntimeError as e: errors.append(str(e))
        t = threading.Thread(target=other); t.start(); t.join(timeout=2)
        self.assertEqual(len(errors), 1)
        self.assertFalse(f.drained)

    def test_every_missing_rank_or_proof_blocks_AC(self):
        required = [('B', 'fenced'), ('A', 'drained'), ('C', 'initial_audit'),
                    ('C', 'computed'), ('C', 'gradient_audit')]
        for missing_role, missing_proof in required:
            for missing_rank in range(8):
                self.p = Ownership(self.cut)
                for role, proof in required:
                    for rank in range(8):
                        if role == missing_role and rank == missing_rank:
                            # Omit its later dependent proofs too.
                            if proof == missing_proof or (role == 'C' and required.index((role, proof)) > required.index((missing_role, missing_proof))):
                                continue
                        self.p.record(self.cut, role, rank, proof)
                with self.subTest(role=missing_role, rank=missing_rank, proof=missing_proof), self.assertRaises(RuntimeError):
                    self.p.choose('AC')

    def test_AC_can_activate_before_B_drain_but_cannot_clean_B(self):
        self.ready(); self.p.choose('AC')
        with self.assertRaises(RuntimeError): self.p.activate('AC')
        self.record('A', 'closed'); self.p.activate('AC')
        for role in 'AC':
            for rank in range(8): self.p.commit(self.cut, role, rank, 7)
        self.assertFalse(self.p.cleanup_allowed('A'))
        self.assertFalse(self.p.cleanup_allowed('B'))
        with self.assertRaises(RuntimeError): self.record('B', 'closed')
        self.record('B', 'drained', 'closed')
        self.assertTrue(self.p.cleanup_allowed('A'))
        self.assertTrue(self.p.cleanup_allowed('B'))

    def test_cancel_timeout_exception_select_new_AB_waits_for_B(self):
        for reason in ('timeout', 'cancel', 'callback_exception'):
            with self.subTest(reason=reason):
                self.p = Ownership(self.cut)
                self.p.choose('AB')
                self.record('A', 'drained')
                with self.assertRaises(RuntimeError): self.record('A', 'closed')
                self.record('B', 'fenced')
                self.record('A', 'closed')
                with self.assertRaises(RuntimeError): self.p.activate('AB')
                self.record('B', 'fenced', 'drained', 'closed')
                self.p.activate('AB')
                self.p.commit(self.cut, 'B', 7, 7)
                with self.assertRaises(RuntimeError): self.p.commit(Cut(2, 6, self.cut.nonce), 'B', 7, 7)

    def test_post_activation_cannot_rollback_even_before_first_commit(self):
        self.ready(); self.record('A', 'closed')
        self.p.choose('AC'); self.p.activate('AC')
        with self.assertRaises(RuntimeError): self.p.choose('AB')
        with self.assertRaises(RuntimeError): self.p.activate('AB')
        with self.assertRaises(RuntimeError): self.p.commit(self.cut, 'B', 0, 7)
        self.p.commit(self.cut, 'C', 0, 7)
        with self.assertRaises(RuntimeError): self.p.commit(self.cut, 'C', 0, 7)
        with self.assertRaises(RuntimeError): self.p.commit(self.cut, 'A', 0, 6)
        with self.assertRaises(RuntimeError): self.p.commit(self.cut, '', 0, 7)

    def test_cancel_cannot_be_reversed_and_bad_proofs_rejected(self):
        self.ready(); self.p.choose('AC'); self.p.choose('AB')
        with self.assertRaises(RuntimeError): self.p.choose('AC')
        for role, rank, proof in [('B', 8, 'fenced'), ('B', True, 'fenced'), ('', 0, 'fenced'), ('AB', 0, 'drained'), ('C', 0, 'closed')]:
            with self.assertRaises(ValueError): self.p.record(self.cut, role, rank, proof)


if __name__ == '__main__': unittest.main()
