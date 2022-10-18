# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .aflw import AFLW
from .cofw import COFW
from .cofw68 import COFW68
from .face300w import Face300W
from .wflw import WFLW
from ._300wlp import _300WLP

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'COFW68', '_300WLP', 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET == 'COFW68':
        return COFW68
    elif config.DATASET.DATASET == '300WLP':
        return _300WLP
    else:
        raise NotImplemented()

