"""Regression: sixteen active done records do not drain a late candidate."""
import unittest
from scripts.v4_transaction_remote import unfinished_actors


class DrainTest(unittest.TestCase):
    def test_late_candidate_is_required_even_after_all_active_ranks_finish(self):
        actors={label:{} for label in ('A','B','C','D','E')}
        rows={label:[dict(kind='done',device=d) for d in range(4)] for label in actors}
        rows['C']=[]
        self.assertEqual(unfinished_actors(actors,rows),['C'])
        rows['C']=[dict(kind='standby_done',device=d) for d in range(3)]
        self.assertEqual(unfinished_actors(actors,rows),['C'])
        rows['C'].append(dict(kind='standby_done',device=3))
        self.assertEqual(unfinished_actors(actors,rows),[])

    def test_duplicate_terminal_marker_cannot_substitute_missing_device(self):
        rows={'C':[dict(kind='standby_done',device=0) for _ in range(4)]}
        self.assertEqual(unfinished_actors({'C':{}},rows),['C'])

    def test_only_explicitly_stopped_owned_actor_can_bypass_drain(self):
        self.assertEqual(unfinished_actors({'B':{'expected_stop':True},'C':{}},{'B':[],'C':[]}),['C'])


if __name__=='__main__':unittest.main()
