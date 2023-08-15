r"""Encoders & Decoders (without probabilistic model)

Constructor:
    Encoder
        arg1: dim_output  (Output shape: (N, dim_output))
        arg2, ... 

    Decoder
        arg1: dim_input   (Input shape: (N, dim_input))
        arg2, ...


References:
    https://github.com/ctallec/world-models/blob/master/models/vae.py
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    https://github.com/dfdazac/vaesbd/blob/master/model.py
"""

# getattr(parts, classname)


from .mobilevit import MobileViTWrap
from .newtonianvae_rep import SpatialBroadcastDecoder, VanillaDecoder, VanillaEncoder
from .parts_backend import (
    DecoderC,
    DecoderCWrap,
    DecoderV1,
    DecoderV2,
    DecoderV3,
    EncoderV1,
    IRBDecoderWrap,
    ResNet,
    ResnetCWrap,
    ResnetDecoder,
    VisualDecoder64,
    VisualDecoder224,
    VisualDecoder224V3,
    VisualDecoder256,
    VisualEncoder64,
    VisualEncoder256,
)
