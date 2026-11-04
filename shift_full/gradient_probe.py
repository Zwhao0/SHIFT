"""Default-off, bounded single-update diagnostic; synchronous immutable CPU slices."""
from types import MethodType


class GradientProbe:
    def __init__(self, torch, dist, stage, buffers, step, progress, metadata):
        self.torch, self.step, self.progress = torch, step, progress
        self.metadata, self.rows, self.selected = metadata, [], []
        names = dict(stage.named_parameters())
        suffixes = ('.self_attn.v_proj.bias', '.mlp.shared_expert_gate.weight')
        chosen = [next((n for n in names if n.endswith(s)), None) for s in suffixes]
        chosen += [next((n for n in names if '.experts.' in n and n.endswith('.gate_proj.weight')), None)]
        if None in chosen or len(set(chosen)) != 3:
            raise RuntimeError('probe representative layout missing')
        for kind, name in enumerate(chosen):
            p = names[name]
            if kind < 2 and p.numel() > 896:
                raise RuntimeError('probe representative exceeds bound')
            indices = list(range(p.numel())) if kind < 2 else sorted({i*(p.numel()-1)//15 for i in range(16)})
            matches = [(i, b) for i, b in enumerate(b for buf in buffers for b in buf.buckets) if any(p is q for q in b.params)]
            if len(matches) != 1:
                raise RuntimeError('probe parameter bucket ambiguous')
            bucket_id, bucket = matches[0]
            info = dict(name=name, shape=list(p.shape), dtype=str(p.main_grad.dtype), indices=indices,
                        bucket=bucket_id, members=dist.get_process_group_ranks(bucket.data_parallel_group),
                        scaling=bucket.gradient_scaling_factor, dense=getattr(p, 'allreduce', True),
                        tensor_parallel=getattr(p, 'tensor_model_parallel', False),
                        partition_dim=getattr(p, 'partition_dim', None))
            self.selected.append((p, bucket, info, torch.tensor(indices, device=p.device)))
        for bucket in {id(b): b for _, b, _, _ in self.selected}.values():
            start, finish = bucket.start_grad_sync, bucket.finish_grad_sync

            def wrapped_start(this, native=start):
                self.capture('pre_local', this)
                return native()

            def wrapped_finish(this, native=finish):
                result = native()  # Native Work.wait establishes current-stream dependency.
                self.capture('post_local', this)
                return result

            bucket.start_grad_sync = MethodType(wrapped_start, bucket)
            bucket.finish_grad_sync = MethodType(wrapped_finish, bucket)

    def capture(self, phase, bucket=None):
        if self.progress['updates'] + 1 != self.step:
            return
        for p, b, info, indices in self.selected:
            if bucket is not None and b is not bucket:
                continue
            if any(r['phase'] == phase and r['name'] == info['name'] for r in self.rows):
                raise RuntimeError('duplicate probe phase')
            # Blocking CPU transfer waits for the original compute/Work.wait stream;
            # retained tensors never alias training storage or consume CUDA RNG.
            values = p.main_grad.detach().reshape(-1).index_select(0, indices).cpu().float().tolist()
            self.rows.append(dict(info, phase=phase, values=values))

    def result(self):
        expected = {(phase, info['name']) for phase in ('pre_local', 'post_local', 'post_outer', 'post_tail')
                    for _, _, info, _ in self.selected}
        if {(r['phase'], r['name']) for r in self.rows} != expected:
            raise RuntimeError('incomplete probe phases')
        return dict(step=self.step, metadata=self.metadata, tensors=self.rows, performance_evidence=False)
