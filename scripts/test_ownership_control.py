import socket
import threading
import unittest
from shift_full.ownership_control import FenceWire
from shift_full.ownership_protocol import Cut,LocalFence


class ControlTest(unittest.TestCase):
    def pair(self):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        a=socket.create_connection(listener.getsockname());b,_=listener.accept();listener.close()
        a,b=FenceWire(a,.1),FenceWire(b,.1)
        self.addCleanup(a.close);self.addCleanup(b.close)
        return a,b

    def test_control_worker_ack_leaves_owner_undrained(self):
        a,b=self.pair();cut=Cut(1,6,'run');f=LocalFence(cut)
        f.reserve(quorum_complete=True,state_readers_done=True)
        t=threading.Thread(target=lambda:b.fence_and_ack(f,3));t.start()
        a.send('fence',cut,3);a.receive('ack',cut,3);t.join(timeout=1)
        self.assertFalse(t.is_alive());self.assertTrue(f.fenced)
        self.assertFalse(f.drained or f.closed)
        with self.assertRaises(RuntimeError):f.allow_old_outer()

    def test_stale_rank_nonce_and_phase_cannot_ack(self):
        for kind,cut,rank in [('ack',Cut(1,6,'n'),3),('fence',Cut(2,6,'n'),3),
                              ('fence',Cut(1,6,'old'),3),('fence',Cut(1,6,'n'),2)]:
            with self.subTest(kind=kind,cut=cut,rank=rank):
                a,b=self.pair();f=LocalFence(Cut(1,6,'n'))
                f.reserve(quorum_complete=True,state_readers_done=True)
                a.send(kind,cut,rank)
                with self.assertRaises(ValueError):b.fence_and_ack(f,3)
                self.assertFalse(f.fenced)

    def test_timeout_eof_and_oversized_packet_reject(self):
        a,b=self.pair()
        with self.assertRaises(socket.timeout):b.receive('fence',Cut(1,6,'n'),0)
        # A timed-out file reader cannot safely be reused: fresh connection.
        a,b=self.pair();a.sock.sendall(b'x'*4097)
        with self.assertRaises(ValueError):b.receive('fence',Cut(1,6,'n'),0)
        a,b=self.pair();a.sock.shutdown(socket.SHUT_WR)
        with self.assertRaises(ValueError):b.receive('fence',Cut(1,6,'n'),0)

if __name__=='__main__':unittest.main()
