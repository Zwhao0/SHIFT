"""Hash compatibility across tensor layouts, chunk boundaries and mutations."""
import unittest
from unittest.mock import patch
import torch
from shift_full.state_digest import digest_reference,digest_staged


class DigestTest(unittest.TestCase):
    def values(self,device):
        return {2:torch.arange(39,dtype=torch.float32,device=device).reshape(3,13).T,
            'state':[(torch.tensor(3.,device=device),torch.empty(0,device=device)),
                {False:torch.arange(17,dtype=torch.bfloat16,device=device),
                 'cpu':torch.tensor([True,False]),'rng':torch.arange(33,dtype=torch.uint8)}],
            'metadata':(None,True,1.5,b'bytes')}

    def check(self,device):
        value=self.values(device);expected=digest_reference(value)
        from shift_full.train import digest as train_digest
        from shift_full.planned_train import digest as planned_digest
        self.assertEqual(expected,train_digest(value))
        self.assertEqual(expected,planned_digest(value))
        with patch.dict('os.environ',{'SHIFT_DIGEST_ENGINE':'staged'}):
            self.assertEqual(expected,train_digest(value))
            self.assertEqual(expected,planned_digest(value))
        for capacity in (1,7,64,1024):self.assertEqual(digest_staged(value,capacity),expected)
        value[2][0,0]+=1
        changed=digest_staged(value,7)
        self.assertNotEqual(changed,expected);self.assertEqual(changed,digest_reference(value))
        self.assertNotEqual(digest_staged([1,2]),digest_staged((1,2)))

    def test_cpu(self):self.check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda(self):self.check('cuda')

    def test_bound(self):
        with self.assertRaises(ValueError):digest_staged({},0)
        from shift_full.state_digest import configured_engine
        with patch.dict('os.environ',{'SHIFT_DIGEST_ENGINE':'invalid'}):
            with self.assertRaises(ValueError):configured_engine()


if __name__=='__main__':unittest.main()
