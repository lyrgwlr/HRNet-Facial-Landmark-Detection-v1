# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

#from .aflw import AFLW
#from .cofw import COFW
#from .face300w import Face300W
#from .wflw import WFLW
from .race import RACE

__all__ = ['RACE', 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'RACE':
        return RACE
    else:
        raise NotImplemented()

