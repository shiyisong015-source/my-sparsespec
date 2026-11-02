"""Verify-history priorities and unique future-page claims; independent of CUDA."""
from dataclasses import dataclass
from numbers import Integral


DISTANCE_SCOPE = "remaining Draft steps to next Verify; Verify=0"


def distance_to_verify(spec_stride, cur_spec_idx, *, verify):
    """Count request progress, including the native initial speculation offset.

    Draft's hot/recent/tail view is protected separately. This distance is
    only for history that needs to be resident at the next Verify.
    """
    if not isinstance(spec_stride, Integral) or isinstance(spec_stride, bool) or spec_stride < 1:
        raise ValueError("Verify distance requires a positive integer stride")
    # Native greedy/naive admission obtains this index from numpy.argmin/max.
    if (not isinstance(cur_spec_idx, Integral) or isinstance(cur_spec_idx, bool)
            or not 0 <= cur_spec_idx <= spec_stride):
        raise ValueError("Speculation index must be within [0, k]")
    return 0 if verify else int(spec_stride - cur_spec_idx)


@dataclass(frozen=True)
class Candidate:
    """A request's Verify-only history; distance is in remaining Draft steps."""
    identity: int
    distance: int
    verify: bool
    waiting_since: int
    last_served: int = -1

    @property
    def restore_key(self):
        return self.distance, self.waiting_since, self.identity

    @property
    def victim_key(self):
        return -self.distance, not self.verify, -self.last_served, self.identity


def may_replace(receiver, victim):
    if receiver.identity == victim.identity:
        return False
    if receiver.distance != victim.distance:
        return receiver.distance < victim.distance
    if receiver.verify != victim.verify:
        return victim.verify  # At equal distance, evict Verify before Draft.
    return receiver.restore_key < victim.restore_key


class FuturePages:
    """Claims name an epoch/chunk producer, never fictitious physical pages."""
    def __init__(self):
        self.claims = {}

    def claim(self, producer, owner, count):
        if count <= 0:
            raise ValueError("Future release must contain pages")
        if producer in self.claims:
            raise ValueError("Future release is already claimed")
        self.claims[producer] = owner, count

    def pages(self, owner):
        return sum(count for target, count in self.claims.values() if target == owner)

    def consume(self, producer, owner):
        if self.claims[producer][0] != owner:
            raise ValueError("Wrong future-page owner")
        return self.claims.pop(producer)[1]

    def cancel(self, owner):
        self.claims = {key: value for key, value in self.claims.items() if value[0] != owner}
