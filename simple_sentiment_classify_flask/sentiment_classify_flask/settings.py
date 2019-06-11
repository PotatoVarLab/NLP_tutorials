#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    :  settings.py
# @Date    :  2019/5/8 11:18
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
MODULE_NAME = 'sentiment_classify_flask'

###########
# Logging #
###########

_LOGGINE_NAME = 'sentiment_classify_flask_main'
_LOGGING_FILE = '/tmp/error_sentiment_classify_flask_main.log'

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
        # 'file': {
        #     'level': 'WARNING',
        #     'class': 'logging.handlers.RotatingFileHandler',
        #     'filename': _LOGGING_FILE,
        #     'maxBytes': 500*1024*1024,
        #     'backupCount': 5,
        #     'formatter': 'file',
        # }
    },
    'loggers': {
        _LOGGINE_NAME: {
            'handlers': ['console'],
            # 'handlers': ['console','file'],
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
# DEBUG_STATUS = True
DEBUG_STATUS = False

#################
# Server Routes #
#################

# short segment sentiment route
SHORT_SEGMENT_SENTIMENT_ROUTE = '/simple_sentiment'

##############
# Others API #
##############

# BERT server API
# BERT_SERVER_API = 'http://192.168.1.104:9000/v1/models/bert_sentiment_clf:predict'
BERT_SERVER_API = 'http://10.0.3.55:9000/v1/models/bert_sentiment_clf:predict'

####################################
# Transform BERT features settings #
####################################

# Input max text length
MAX_TEXT_LENGTH = 1024

# Label List
LABEL_LIST = ['0', '1', '2']

# BERT max seq length
MAX_SEQ_LENGTH = 128

# BERT positive threshold score
POSITIVE_THRESHOLD_SCORE = 0.998

# BERT negative threshold score
NEGATIVE_THRESHOLD_SCORE = 0.90