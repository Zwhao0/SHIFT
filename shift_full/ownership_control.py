"""Small independent CPU control connection; never imports Torch or Manager.

Connections are prepared off the critical path. One ticket per reserved cut;
the caller owns bounded socket/thread teardown and all training work.
"""
import json
import socket
from dataclasses import asdict


class FenceWire:
    def __init__(self, sock, timeout=5):
        self.sock=sock
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.reader=sock.makefile('rb')

    def send(self, kind, cut, rank):
        self.sock.sendall((json.dumps(dict(kind=kind,rank=rank,cut=asdict(cut)))+'\n').encode())

    def receive(self, kind, cut, rank):
        line=self.reader.readline(4097)
        if len(line)>4096 or not line.endswith(b'\n'):
            raise ValueError('incomplete/oversized control packet')
        value=json.loads(line)
        if value!=dict(kind=kind,rank=rank,cut=asdict(cut)):
            raise ValueError('wrong control ticket/rank/phase')

    def fence_and_ack(self, local_fence, rank):
        cut=local_fence.cut
        self.receive('fence',cut,rank)
        local_fence.control_fence(cut)
        self.send('ack',cut,rank)

    def close(self):
        try:self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:pass
        self.reader.close()
        self.sock.close()
