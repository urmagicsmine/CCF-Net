from .generalized_attention import GeneralizedAttention
from .non_local import NonLocal2D
from .asy_non_local import AsyNonLocal2D
from .cbam import CBAM
#from .crisscross_attention import CCAttention # need mmcv-full
#from .cc_attention import CrissCrossAttention, ca_weight, ca_map
#from .dcn_align import Align_2D, Align_3D
from .afnb import SelfAttentionBlock2D, AFNB

CCAttention=None
CrissCrossAttention=ca_weight=ca_map=None
Align_2D=Align_3D=None

__all__ = ['NonLocal2D', 'GeneralizedAttention', 'CBAM',
        'CrissCrossAttention', 'ca_weight', 'ca_map',
        'CCAttention',
        'Align_2D', 'Align_3D',
        'AsyNonLocal2D', 'SelfAttentionBlock2D', 'AFNB']
