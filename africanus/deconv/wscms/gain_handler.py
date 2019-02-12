#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ...util.docs import DocstringTemplate

import numba
import numpy as np


def set_scale_gain(convpsf, convpsf0, volumes, iscale, gain):
    return gain * convpsf0.max() * volumes[iscale] / (convpsf.max() * volumes[0])
