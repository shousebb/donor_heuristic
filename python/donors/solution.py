"""The result every solver returns."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Solution:
    """A donor selection; tau and lower are in integer units of 1 / scale.

    `sets` holds distinct donor sets at the same tau, fewest parents at tau first,
    with their `counts`; sets[0] is `donors`.
    """

    donors: list[int]
    tau: int
    count: int  # parents at tau
    lower: int | None  # proven lower bound on tau; None for heuristics
    seconds: float
    status: str
    sets: list[list[int]]
    counts: list[int]

    def summary(self, name: str, scale: int) -> str:
        """One line in distance units."""
        lower = "" if self.lower is None else f" lower bound={self.lower / scale:.4f}"
        return (
            f"{name}: status={self.status} tau={self.tau / scale:.4f}{lower} "
            f"time={self.seconds:.1f}s"
        )
