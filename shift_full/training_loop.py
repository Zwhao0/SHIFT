"""Policy-driven synchronous loop around the shared training transaction."""

from .training_core import complete_step


def _close_unresolved(transaction):
    close = getattr(transaction, 'close', None)
    if close is not None:
        try:
            close()
        except BaseException:
            pass


def _nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def run_loop(runtime, policy, *, until=None, max_attempts=None):
    """Run attempts until committed updates reach ``until`` or the worker stops.

    A policy selected at a boundary owns the whole attempt, including its vote
    and commit/skip notification. Policy changes during an attempt take effect
    at the next boundary.
    """
    if until is not None:
        _nonnegative_int('until', until)
    if max_attempts is not None:
        _nonnegative_int('max_attempts', max_attempts)

    if getattr(runtime, 'policy', None) is None:
        runtime.policy = policy

    attempts = 0
    while until is None or runtime.progress['updates'] < until:
        boundary_policy = runtime.policy
        if boundary_policy.before_iteration(runtime) is False:
            break
        if runtime.should_stop():
            break
        if until is not None and runtime.progress['updates'] >= until:
            break
        if max_attempts is not None and attempts >= max_attempts:
            raise RuntimeError(
                f'max_attempts={max_attempts} reached with '
                f'updates={runtime.progress["updates"]}; until={until} not reached'
            )

        attempt_policy = runtime.policy
        attempts += 1
        context = attempt_policy.begin(runtime)
        transaction = runtime.compute(context)
        resolved = False
        try:
            try:
                computed = next(transaction)
            except StopIteration as exc:
                raise RuntimeError('runtime.compute(context) must yield a computed value') from exc
            if not hasattr(computed, 'finite'):
                raise TypeError('computed value must expose a finite attribute')

            commit = attempt_policy.vote(runtime, context, computed)
            if type(commit) is not bool:raise TypeError('policy vote must be a boolean')
            update = complete_step(transaction, commit)
            resolved = True
        except BaseException:
            if not resolved:
                _close_unresolved(transaction)
            raise

        if commit:
            runtime.committed(context, computed, update)
        else:
            runtime.skipped(context, computed)

    return runtime.progress['updates']
