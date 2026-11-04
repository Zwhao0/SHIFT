import copy
import unittest
from shift_full.verify_v3_recovery import modern_checks,FIELDS


class ModernRecoveryTest(unittest.TestCase):
    def evidence(self):
        hashes={k:'same' for k in FIELDS};roles={}
        for rid,generation in ((0,0),(1,0),(1,1)):
            for rank in range(8):
                events=[dict(kind='ready',state_schema=2,hashes=hashes)]
                if (rid,generation)==(1,1):events.append(dict(kind='restored',step=4,expected_hashes=hashes,actual_hashes=hashes,hashes_match=True))
                if rid==0:events.append(dict(kind='snapshot_readonly_verified',step=4,tensors=12))
                if rid==0 or generation==1:
                    events += [dict(kind='commit',step=k,participants=2,contributed=True,finite_gradients=True,hashes=hashes if k==5 else {}) for k in range(5,15)]
                roles[rid,generation,rank]=events
        fault=dict(target=1,generation=1,restored_step=4,first_joint_step=5,backup_mode='cold',standby_resources=[])
        return roles,[fault]

    def test_modern_complete(self):
        roles,faults=self.evidence();self.assertEqual(modern_checks({},roles,faults),[])

    def test_old_hashes_cannot_claim_complete_recovery(self):
        roles,faults=self.evidence();roles=copy.deepcopy(roles)
        del roles[1,1,0][0]['hashes']['buffers']
        self.assertTrue(modern_checks({},roles,faults))

    def test_sender_guard_and_tenth_joint_are_required(self):
        roles,faults=self.evidence();roles[0,0,0]=[e for e in roles[0,0,0] if e['kind']!='snapshot_readonly_verified']
        roles[1,1,7].pop()
        errors=modern_checks({},roles,faults)
        self.assertTrue(any('read-only' in e for e in errors));self.assertTrue(any('step14' in e for e in errors))


if __name__=='__main__':unittest.main()
