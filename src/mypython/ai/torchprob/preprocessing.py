"""
References:
    sklearn.preprocessing
"""

from typing import TypeVar


class Scaler:
    """
    x' = (x - a) / b
    """

    T = TypeVar("T")

    def __init__(self, a=0, b=1) -> None:
        assert b > 0

        self.a = a
        self.b = b

    def pre(self, x: T) -> T:
        return (x - self.a) / self.b

    def post(self, x_prime: T) -> T:
        """Applicable to sampled data or loc (mean)
        E[X] = bE[(X-a)/b] + a
        """
        return self.b * x_prime + self.a

    def post_scale(self, scale: T) -> T:
        """V[X] = b^2V[(X-a)/b]"""
        return self.b**2 * scale

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
            + ")"
        )
