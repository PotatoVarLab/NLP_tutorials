#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    :  settings.py
# @Date    :  2019/4/2 10:30
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
import logging.config


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

###############
# Module Info #
###############
MODULE_NAME = 'hot_word_match'

###########
# Logging #
###########

_LOGGINE_NAME = 'hot_word_match_main'
_LOGGING_FILE = '/tmp/error_hot_word_match_main.log'

# Custom logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    # 'filters': {
    #     'require_debug_false': {
    #         '()': '',
    #     },
    #     'require_debug_true': {
    #         '()': '',
    #     },
    # },
    'formatters': {
        'console': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s\t%(filename)s[line:%(lineno)d]\t%(levelname)s\t%(message)s',
        },
        'file': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s\tpid:%(process)d\t%(filename)s[line:%(lineno)d]\t%(levelname)s\t%(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'console',
        },
        'file': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': _LOGGING_FILE,
            'maxBytes': 500*1024*1024,
            'backupCount': 5,
            'formatter': 'file',
        }
    },
    'loggers': {
        _LOGGINE_NAME: {
            # 'handlers': ['console'],
            'handlers': ['console','file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    }
}

logging.config.dictConfig(LOGGING)

################
# DEBUG STATUS #
################

# Debug status of this project
DEBUG_STATUS = True





