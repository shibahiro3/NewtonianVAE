"""
Ref:
    https://github.com/masa-su/pixyz
    https://github.com/pyro-ppl/pyro
"""

from .distributions.bernoulli import Bernoulli

# from .distributions.gmm import GMM
from .distributions.normal import Normal, Normal01
from .functions import KLdiv, log
