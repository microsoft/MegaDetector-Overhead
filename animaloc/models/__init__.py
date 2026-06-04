__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


from .register import MODELS

from .faster_rcnn import *
from .dla import *
from .herdnet import *
from .utils import *
from .ss_dla import *
from .herdnet_detection_branch import *
from .herdnet_hybrid_multiscale_residual import *
from .herdnet_dinov2 import *

__all__ = ['MODELS', *MODELS.registry_names]