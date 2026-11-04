"""CPU-only tests for epoch fencing and four-candidate replacement decisions."""
import unittest

from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.v4_transaction import OwnerGate, Transaction


RANKS=(4,5,6,7)


def hashes(seed):
    return {key:format(seed+index,'064x') for index,key in enumerate(sorted(REFERENCE_HASH_KEYS))}


def approve_all(transaction, epoch, ranks=RANKS):
    for rank in ranks:
        value=hashes(epoch*100+rank)
        if transaction.candidate(epoch,rank,value,dict(value)) is not True:
            raise AssertionError(f'valid candidate rejected: {rank}')


class OwnerGateTest(unittest.TestCase):
    def test_two_rounds_switch_b_to_c_then_c_to_b(self):
        gate=OwnerGate(epoch=0,owner='B')
        self.assertIsNone(gate.authorize('A',0))
        self.assertIsNone(gate.authorize('B',0))

        first=Transaction(1,'B','C',cut=12,ranks=RANKS)
        approve_all(first,1)
        self.assertEqual(first.decide(),'replace')
        gate.fence()
        with self.assertRaises(ValueError):gate.authorize('B',0)
        first.authorize_target(1)
        gate.advance(1,'C')
        gate.authorize('A',1)
        gate.authorize('C',1)
        with self.assertRaises(ValueError):gate.authorize('B',1)

        second=Transaction(2,'C','B',cut=24,ranks=RANKS)
        approve_all(second,2)
        self.assertEqual(second.decide(),'replace')
        gate.fence()
        with self.assertRaises(ValueError):gate.authorize('C',1)
        second.authorize_target(2)
        gate.advance(2,'B')
        gate.authorize('A',2)
        gate.authorize('B',2)
        with self.assertRaises(ValueError):gate.authorize('C',2)

    def test_source_fence_rejects_old_epoch_commit(self):
        gate=OwnerGate(epoch=3,owner='B')
        gate.fence()
        with self.assertRaises(ValueError):gate.authorize('B',3)
        gate.advance(4,'C')
        with self.assertRaises(ValueError):gate.authorize('B',3)

    def test_cancel_advances_epoch_with_b_retained(self):
        gate=OwnerGate(epoch=0,owner='B')
        tx=Transaction(1,'B','C',cut=8,ranks=RANKS)
        self.assertEqual(tx.decide(cancel=True),'retain')
        gate.fence()
        with self.assertRaises(ValueError):gate.authorize('B',0)
        with self.assertRaises(ValueError):tx.authorize_target(1)
        gate.advance(1,'B')
        gate.authorize('A',1)
        gate.authorize('B',1)
        with self.assertRaises(ValueError):gate.authorize('C',1)
        with self.assertRaises(ValueError):gate.authorize('B',0)

    def test_gate_rejects_float_bool_and_nonmonotonic_epochs(self):
        with self.assertRaises(ValueError):OwnerGate(epoch=True)
        with self.assertRaises(ValueError):OwnerGate(epoch=1.0)
        gate=OwnerGate()
        for role,epoch in (('B',True),('B',0.0),('C',0)):
            with self.subTest(role=role,epoch=epoch),self.assertRaises(ValueError):
                gate.authorize(role,epoch)
        gate.fence()
        for epoch,owner in ((True,'C'),(1.0,'C'),(2,'C')):
            with self.subTest(epoch=epoch),self.assertRaises(ValueError):
                gate.advance(epoch,owner)
        gate.advance(1,'C')
        with self.assertRaises(ValueError):gate.advance(2.0,'B')


class TransactionTest(unittest.TestCase):
    def transaction(self):
        return Transaction(1,'B','C',cut=12,ranks=RANKS)

    def test_all_four_complete_nine_hash_reports_are_required_to_replace(self):
        tx=self.transaction()
        for rank in RANKS[:3]:
            value=hashes(100+rank)
            self.assertTrue(tx.candidate(1,rank,value,dict(value)))
        with self.assertRaises(ValueError):tx.decide()
        rank=RANKS[3];value=hashes(100+rank)
        self.assertTrue(tx.candidate(1,rank,value,dict(value)))
        self.assertEqual(tx.decide(),'replace')
        self.assertIsNone(tx.authorize_target(1))

    def test_missing_or_wrong_hash_candidate_forces_retain(self):
        tx=self.transaction()
        for index,rank in enumerate(RANKS):
            actual=hashes(200+rank);expected=dict(actual)
            if index==0:
                actual.pop(next(iter(actual)))
            elif index==1:
                actual[next(iter(actual))]='f'*64
            accepted=tx.candidate(1,rank,actual,expected)
            self.assertEqual(accepted,index>=2)
        self.assertEqual(tx.decide(),'retain')
        with self.assertRaises(ValueError):tx.authorize_target(1)

    def test_bad_hash_encoding_cannot_replace(self):
        tx=self.transaction()
        for index,rank in enumerate(RANKS):
            actual=hashes(300+rank);expected=dict(actual)
            if index==0:actual[next(iter(actual))]='a'*63
            if index==1:actual[next(iter(actual))]='G'*64
            accepted=tx.candidate(1,rank,actual,expected)
            self.assertEqual(accepted,index>=2)
        self.assertEqual(tx.decide(),'retain')

    def test_missing_report_cannot_decide_but_cancel_can(self):
        tx=self.transaction()
        with self.assertRaises(ValueError):tx.decide()
        self.assertEqual(tx.decide(cancel=True),'retain')
        with self.assertRaises(ValueError):tx.authorize_target(1)

    def test_duplicate_late_stale_float_and_bool_candidates_are_rejected(self):
        tx=self.transaction();value=hashes(400)
        self.assertTrue(tx.candidate(1,RANKS[0],value,dict(value)))
        with self.assertRaises(ValueError):tx.candidate(1,RANKS[0],value,dict(value))
        for epoch,rank in ((0,RANKS[1]),(1.0,RANKS[1]),(True,RANKS[1]),
                           (1,4.0),(1,True)):
            with self.subTest(epoch=epoch,rank=rank),self.assertRaises(ValueError):
                tx.candidate(epoch,rank,value,dict(value))
        for epoch in (1.0,True,0):
            with self.subTest(target_epoch=epoch),self.assertRaises(ValueError):
                tx.authorize_target(epoch)
        for epoch in (0,True,1.0):
            with self.subTest(decision_epoch=epoch),self.assertRaises(ValueError):
                Transaction(epoch,'B','C',cut=1,ranks=RANKS)
        for rank_set in ((4,5,6,7.0),(4,5,6,True),(4,5,6,8)):
            with self.subTest(ranks=rank_set),self.assertRaises(ValueError):
                Transaction(1,'B','C',cut=1,ranks=rank_set)

    def test_decision_is_final_and_cancelled_transaction_cannot_reactivate(self):
        tx=self.transaction();approve_all(tx,1)
        self.assertEqual(tx.decide(),'replace')
        with self.assertRaises(ValueError):tx.decide(cancel=True)
        with self.assertRaises(ValueError):tx.candidate(1,RANKS[0],hashes(500),hashes(500))
        with self.assertRaises(ValueError):tx.authorize_target(2)

        cancelled=self.transaction()
        self.assertEqual(cancelled.decide(cancel=True),'retain')
        with self.assertRaises(ValueError):cancelled.decide()
        with self.assertRaises(ValueError):cancelled.authorize_target(1)
        with self.assertRaises(ValueError):cancelled.candidate(1,RANKS[0],hashes(600),hashes(600))


if __name__=='__main__':
    unittest.main()
