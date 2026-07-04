import math


def perplexity(avg_loss: float) -> float:
    try:
        return math.exp(avg_loss)
    except OverflowError:
        return float("inf")


class RunningAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)