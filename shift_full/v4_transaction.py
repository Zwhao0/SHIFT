"""Epoch fencing and all-candidate validation for repeatable transparent switches."""
from .reference_audit import REFERENCE_HASH_KEYS


class OwnerGate:
    def __init__(self,epoch=0,owner='B'):
        if type(epoch) is not int or epoch<0 or owner not in ('B','C'):
            raise ValueError('nonnegative epoch and B/C owner required')
        self.epoch,self.owner,self.fenced=epoch,owner,False

    def authorize(self,role,epoch):
        if self.fenced or type(epoch) is not int or epoch!=self.epoch or role not in ('A',self.owner):
            raise ValueError('stale, fenced or inactive owner')

    def fence(self):
        if self.fenced:raise ValueError('already fenced')
        self.fenced=True

    def advance(self,epoch,owner):
        if type(epoch) is not int or epoch!=self.epoch+1 or owner not in ('B','C'):
            raise ValueError('next monotonic epoch and valid owner required')
        self.epoch,self.owner,self.fenced=epoch,owner,False


class Transaction:
    def __init__(self,epoch,source,target,cut,ranks):
        if (type(epoch) is not int or epoch<1 or source not in ('B','C') or
            target not in ('B','C') or source==target or type(cut) is not int or cut<1):
            raise ValueError('invalid switch boundary')
        ranks=tuple(ranks)
        if len(ranks)!=4 or len(set(ranks))!=4 or any(type(r) is not int or r not in range(8) for r in ranks):
            raise ValueError('four distinct logical candidate ranks required')
        self.epoch,self.source,self.target,self.cut,self.ranks=epoch,source,target,cut,ranks
        self.results={};self.decision=None

    def candidate(self,epoch,rank,actual,expected):
        if (type(epoch) is not int or epoch!=self.epoch or type(rank) is not int or
            self.decision is not None or rank not in self.ranks or rank in self.results):
            raise ValueError('stale, duplicate or unexpected candidate result')
        valid=(isinstance(actual,dict) and isinstance(expected,dict) and
               set(actual)==set(expected)==REFERENCE_HASH_KEYS and actual==expected and
               all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in actual.values()))
        self.results[rank]=valid
        return valid

    def decide(self,*,cancel=False):
        if type(cancel) is not bool:raise ValueError('cancel must be boolean')
        if self.decision is not None:raise ValueError('switch decision is already final')
        if not cancel and len(self.results)!=4:raise ValueError('all candidates must be validated')
        self.decision='retain' if cancel or not all(self.results.values()) else 'replace'
        return self.decision

    def authorize_target(self,epoch):
        if type(epoch) is not int or epoch!=self.epoch or self.decision!='replace':raise ValueError('candidate cannot activate')
