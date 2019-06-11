#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    :  local_server_tes.py
# @Date    :  2019/5/8 12:19
# @Author  :  H.wei
# @Version :  1.0
# @License :  Copyright (C) 2019 YPLSEC, Inc.
# 
# @Desc    :  None

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

from sentiment_classify_flask.sentiment_classify_flask_manager import app

app.run(host='0.0.0.0', port=9090, debug=True)

