"""
This package just keeps the parameters of the probability distribution in the instance.
This will enable coding that is similar to a formula.

References:
    https://github.com/masa-su/pixyz
    https://github.com/pyro-ppl/pyro
"""

from . import config
from .distributions.base import ProbParamsValueError
from .distributions.bernoulli import Bernoulli
from .distributions.gmm import GMM
from .distributions.normal import Normal, Normal01
from .functions import KLdiv, log
