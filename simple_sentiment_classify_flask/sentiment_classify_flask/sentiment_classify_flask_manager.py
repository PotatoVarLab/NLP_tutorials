#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    :  sentiment_classify_flask_manager.py
# @Date    :  2019/5/29 18:20
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
import re
import logging
import json
import demjson
import requests
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime

from .settings import (_LOGGINE_NAME,
                       BASE_DIR,
                       DEBUG_STATUS,
                       SHORT_SEGMENT_SENTIMENT_ROUTE,
                       BERT_SERVER_API,
                       MAX_SEQ_LENGTH,
                       MAX_TEXT_LENGTH,
                       POSITIVE_THRESHOLD_SCORE,
                       NEGATIVE_THRESHOLD_SCORE)


logger = logging.getLogger(_LOGGINE_NAME)


class HotWordsMatchWrapClass(object):
    """The wrapper class for hot-word match,need other package to handle."""
    def __init__(self):
        try:
            from .hot_word_pkg.hot_word_match_manager import HotWordMatchHandler
            self._hot_word_handler = HotWordMatchHandler()
        except ImportError as e:
            logger.exception(e)
            self._hot_word_handler = None

    def recall_base_func(self, content):
        res_info = {}
        if hasattr(self._hot_word_handler, 'analysis'):
            try:
                res_info = self._hot_word_handler.analysis(content)
            except Exception as e:
                logger.exception(e)

        return res_info

def requests_BERT_server(feature):
    emotion_keys = ['pos', 'neg', 'neu']
    probs = [0.023, 0.2, 0.777]
    try:
        #response = requests.post(BERT_SERVER_API, json=feature, timeout=2)
        response = requests.post(BERT_SERVER_API, json=feature)
        if response.status_code == 200:
            pred = demjson.decode(response.content)
            #probs = np.array(pred['predictions'])[0]
            probs = np.array(pred['outputs'])[0]
            logger.debug('BERT server result: {}'.format(probs))
    except Exception as e:
        logger.exception(e)

    sentiment_result = dict(zip(emotion_keys, probs))
    result = ['neu', 0.667]
    if sentiment_result and isinstance(sentiment_result, dict):
        try:
            sentiment_result = sorted(sentiment_result.items(), key=lambda d: d[1], reverse=True)
            label, prob = sentiment_result[0]
            prob = float(prob)
            if label == 'pos':
                if prob >= POSITIVE_THRESHOLD_SCORE:
                    label = 'pos'
                else:
                    label = 'neu'
                    prob = prob if 0.6 <= prob <= 1.0 else 0.777
            elif label == 'neg':
                if prob >= NEGATIVE_THRESHOLD_SCORE:
                    label = 'neg'
                else:
                    label = 'neu'
                    prob = prob if 0.6 <= prob <= 1.0 else 0.777
            else:
                label = 'neu'

            result = [label, prob]
        except Exception as e:
            pass
    logger.debug('Finally result: {}'.format(result))

    return result

def title_school_sensitive_match(title):
    school_keys = ["学校","大学","校长","学院","院系","我校","教师","教授","校党委","教职工","教导主任","班主任","辅导员","校党委","校团委","研究生","博士","室友","校友","校园欺凌","校园暴力","学校门事件"]

    sensitive_event_keys = ['不力','黑恶','涉嫌', '违规','违纪','违法','贪污','受贿','行贿', '被查', '被"双开"','被双开','被调查','被审查','被撤销','被处分','被调岗','被公安机关','行政拘留','玩忽职守','管理混乱','形式主义','官僚主义','生活作风','通报批评','滥用职权','违反中央','涉嫌犯罪','立案查处','审查调查','造成不良','思想麻痹','留党察看','开除党籍','开除公职','撤职处分','行政警告处分','党内警告处分','党纪政纪处分','行政记过处分','利用职务之便','职务上的便利','为他人谋取利益','被依法追究刑事责任','事故原因','校园欺凌','学校门事件','校园意外事故','校园突发事件','暴力','死伤','火灾','爆炸','跳楼','猝死','失踪','自虐','自杀','上访','绑架','中毒','投毒','犯罪','行凶','殴打','斗殴','重大伤害','伤害事故','重大事故','突发事件','恶性事件','恶性行为','失控行为','群体性事件','报复性行为','传染性疾病','传染病','学生不幸','经抢救无效死亡','隐瞒事实真相', '冲突', '引发社会']

    school_sensitive_flag = False
    try:
        logger.debug('--> school func input-title: {}'.format(title))
        for k in school_keys:
            school_match = re.search(k, title)
            if school_match:
                for ek in sensitive_event_keys:
                    senti_match = re.search(ek, title)
                    if senti_match:
                        school_sensitive_flag = True
                        logger.debug('School sensitive matched: {} - {}'.format(school_match.group(), senti_match.group()))
                        break
            if school_sensitive_flag:
                break
    except Exception as e:
        pass

    return school_sensitive_flag



### try import other pkg
try:
    from .transform_bert_features import TransformBERTFeaturesHandler
    trans_bert_features_handler = TransformBERTFeaturesHandler()
    from .bert_segment_clear_up import BERTClearUpSegmentBase
    bert_clear_up_segment_handler = BERTClearUpSegmentBase()

    hot_word_match_init_handler = HotWordsMatchWrapClass()
except ImportError as e:
    print(e)
    trans_bert_features_handler = None
############
# Init app #
############
app = Flask(__name__)
app.debug = DEBUG_STATUS


# less than 6 words need match key-words
LESS_6_WORDS_KEYS = ["杀", "死", "毙", "灭", "艹", "操", "渣", "恶", "毒", "疼", "痛", "打", "砸", "缺", "黑", "肿", "奸", "哀", "悲", "政", "税", "血", "尼", "贱", "匪", "霸", "日", "病", "心", "恨", "逼", "毒", "赌", "滥", "惨", "裸", "瞎", "残", "怒", "粗", "洗", "崩", "溃", "官", "猝", "怜", "邪", "伪", "拆", "滚", "自", "失", "暴", "不", "差", "无", "破", "罪", "被", "诛", "苦", "特么", "丫的", "sb", "2b", "tm", "tmd", "nnd", "妈的", "脑残", "傻x", "人肉", "可怕", "腐败", "天朝", "畜生", "禽兽", "流氓", "郁闷", "藏独", "台独", "疆独", "东突", "国家", "政府", "体制", "城管", "国人", "抵制", "无耻", "变态", "传销", "气愤", "mlgb", "王八蛋", "天理", "共军", "共党", "走好", "垃圾", "上访"]

# Testing URL
@app.route('/hello', methods=['GET', 'POST'])
def hello_test():
    return 'Hello, This is sentiment classify server test URL!'


# Simple sentiment classify URL
@app.route(SHORT_SEGMENT_SENTIMENT_ROUTE, methods=['POST'])
def short_segment_sentiment_classify():
    default_result = {
        "active_words": "",
        "negative_words": "",
        "neutral_words": "",
        "political_words": "",
        "school_words": "",
        "related_school": 0,
        "remarks": "",
        "emotion_polarity": "neu",
        "score": 0.777,
    }
    try:
        ## parse args
        data = request.get_data(cache=False, as_text=True, parse_form_data=True)
        # print(type(data),data)
        ## recall some models
        if isinstance(data, str) and data.strip():
            meta_dict = demjson.decode(data)
            # print(meta_dict)
            dtype = meta_dict.get('dtype', '')
            title = meta_dict.get('title', '')
            content = meta_dict.get('content', '')
            if title or content:
                segment = title + content
            else:
                segment = ''
            ##====== Hot word match =======##
            if segment:
                max_segment = segment[:MAX_TEXT_LENGTH]
                st = datetime.now()
                hot_word_match_result = hot_word_match_init_handler.recall_base_func(max_segment)
                # hot_word_match_result = ''
                if hot_word_match_result and isinstance(hot_word_match_result, dict):
                    default_result['active_words'] = hot_word_match_result.get('active_words', '')
                    default_result['negative_words'] = hot_word_match_result.get('negative_words', '')
                    default_result['neutral_words'] = hot_word_match_result.get('neutral_words', '')
                    default_result['political_words'] = hot_word_match_result.get('political_words', '')
                    default_result['school_words'] = hot_word_match_result.get('school_words', '')
                    default_result['related_school'] = hot_word_match_result.get('related_school', 0)
                    default_result['remarks'] = hot_word_match_result.get('remarks', '')
                print('==> hot word match use time: {}'.format(datetime.now() - st))

                ##====== school sensitive title match ====##
                school_sensi_match_flag = title_school_sensitive_match(title)
                if school_sensi_match_flag:
                    default_result['emotion_polarity'] = 'neg'
                    default_result['score'] = 0.933333

                ##====== BERT server sentiment =======##
                if not school_sensi_match_flag and hasattr(trans_bert_features_handler, 'transform_bert_feature_batch') \
                        and hasattr(bert_clear_up_segment_handler, 'clear_up_str'):
                    st1 = datetime.now()
                    bert_segment = bert_clear_up_segment_handler.clear_up_str(max_segment)[:MAX_SEQ_LENGTH]
                    logger.debug('Input bert seg: {}'.format(bert_segment))

                    bert_features = trans_bert_features_handler.transform_bert_feature_batch(bert_segment)
                    print('==> feature transform use time: {}'.format(datetime.now() - st1))
                    payload = {
                        #"instances": [bert_features]
                        "inputs": bert_features
                    }
                    #logger.debug('Input features: {}'.format(bert_features))
                    st2 = datetime.now()
                    bert_sentiment_result = requests_BERT_server(payload)
                    #==== less than 6 words need match ====#
                    #if len(bert_segment) <= 6:
                    if len(re.findall(r'[\u4e00-\u9fa5]', bert_segment)) <= 6:
                        for i in LESS_6_WORDS_KEYS:
                            if re.search(i, bert_segment):
                                default_result['emotion_polarity'] = bert_sentiment_result[0]
                                default_result['score'] = bert_sentiment_result[1]
                                break
                    else:
                        default_result['emotion_polarity'] = bert_sentiment_result[0]
                        default_result['score'] = bert_sentiment_result[1]
                    print('==> bert server use time: {}'.format(datetime.now() - st2))
            else:
                logger.debug('Segment has nothing.')
        else:
            logger.warning('Data do not match analysis conditions.')
    except Exception as e:
        logger.exception(e)

    ## clear up results
    return jsonify(default_result)



