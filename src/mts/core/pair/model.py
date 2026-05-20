from dataclasses import dataclass


@dataclass(init=False)
class DistancePair:
    st: str
    nd: str
    distance: float

    def __init__(
        self,
        st: str,
        nd: str,
        distance: float,
    ) -> None:
        self.st, self.nd = sorted((st, nd))
        self.distance = distance
