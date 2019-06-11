#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : weibo_sensitive_redis.py
# @Date    : 2018/2/13 16:02
# @Author  : Huizi
# @Version :
# Copyright 2018-present YPLSEC, Inc.

"""
description:weibo 文本处理模块

"""

import re
import os
import sys
import json
import logging
import traceback

# import pkuseg
import jieba
import pickle
import numpy as np

from .settings import BASE_DIR, _LOGGINE_NAME, MODULE_NAME, DEBUG_STATUS

# import fasttext

logger = logging.getLogger(_LOGGINE_NAME)


class HotWordMatchHandler(object):
    def __init__(self):
        self.module_name = MODULE_NAME
        self.debug = DEBUG_STATUS
        self.loaddic = self.loaddicts()
        self.configs = self.get_ana_info()
        self.model = os.path.join(BASE_DIR, 'data/fasttext_model.bin')
        self.xuexiao_count = self.load_moduel(os.path.join(BASE_DIR, 'data/pkl/', 'xuexiao_count.pkl'))
        self.xuexiao_pv = self.load_moduel(os.path.join(BASE_DIR, 'data/pkl/', 'xuexiao_pv.pkl'))
        # self.seg = pkuseg.pkuseg(model_name=os.path.join(BASE_DIR, 'weibo'),user_dict=os.path.join(BASE_DIR, 'data/selfdict.txt'))
        jieba.load_userdict(os.path.join(BASE_DIR, 'data/selfdict.txt'))

    def load_moduel(self, modelname):
        # model_path = os.path.join(BASE_DIR, 'data/pkl/')
        # model_path = './data/pkl/'
        with open(modelname, "rb") as ft:
            model = pickle.load(ft, encoding='iso-8859-1')
        return model

    def getmd5_txt(self):
        md5list = []
        with open(os.path.join(BASE_DIR, 'data/weibo_md5.txt'), 'r') as ft:
            lines = ft.readlines()
            for line in lines:
                line.strip('\n')
                md5list.append(line)
        return md5list

    def get_ana_info(self):
        ''' ：info_dir 保存模型配置文件的文件夹路径
        '''
        with open(os.path.join(BASE_DIR, 'data/config.txt')) as game_info:
            # sprint game_info.read()
            radata = json.load(game_info, encoding="utf-8")
        return radata[0]

    def delherf(self, content):
        re_urlicon = re.compile('<img [^>].* src="[^"].*timeline_card_small_location_default.png">[\s\S]*')
        content = re_urlicon.sub('', content)
        re_http = re.compile('http://.*(?:jpg|gif)')
        content = re_http.sub('', content)
        re_c = re.compile('saimowfrv48v5dlsr1057i8')
        content = re_c.sub('', content)
        re_a = re.findall(r'@([\u4e00-\u9fa5A-Z0-9a-z._-]+)', content)
        yonghu = re_a
        re_a = re.compile('@([\u4e00-\u9fa5A-Z0-9a-z._-]+)')
        content = re_a.sub('', content)
        re_href = re.compile('</?a[^>]*>')
        content = re_href.sub('', content)
        re_em = re.compile('</?(em|span|i|p|div)[^>]*>')
        content = re_em.sub('', content)
        re_img = re.compile('</?img[^>]*>')
        content = re_img.sub('', content)
        # re_wbicon = re.compile('<[^>]*>')
        # re_wbicon = re.compile('<i class="wbicon">[^>]*>.*')
        re_wbicon = re.compile('class="wbicon"[^>]*>[\s\S]*')
        content = re_wbicon.sub('', content)
        re_br = re.compile('<br/>')
        content = re_br.sub('', content)
        # biaoqing = list(set(re.findall(r'\[(.*?)\]', content)))
        # print(biaoqing)
        re_biaoqing = re.compile(r'\[(.*?)\]')
        content = re_biaoqing.sub('', content)
        content = re.sub('\u200b', '', content)
        content = re.sub('\ufeff', '', content)
        re_huati = re.findall(r'#(.*?)#', content)  # 去掉话题内内容
        huati = re_huati
        content = re.sub(r'#(.*?)#', "", content)
        return content, yonghu, huati

    def stopwordslist(self, filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    def cutword(self, weibo_content):
        words = []
        # cutword = self.seg.cut(weibo_content)
        cutword = jieba.lcut(weibo_content)
        stopwords = self.stopwordslist(os.path.join(BASE_DIR, 'data/stopwords.txt'))
        for word in cutword:
            if word not in stopwords:
                if word != '\t':
                    words.append(word)
        outstr = " ".join(words)
        words = set(words)
        return outstr, words

    def changewords(self, setwords):
        data = []
        words = setwords
        metedata = " ".join(list(words))
        data.append(metedata)
        return data

    def gethuati(self, huati):
        huati_cut = []
        for i in huati:
            outstr, words = self.cutword(i)
            huati_cut.extend(words)
        data_huati = self.changewords(set(huati_cut))
        return data_huati

    def getyonghu(self, yonghu):
        yonghu_cut = []
        for i in yonghu:
            outstr, words = self.cutword(i)
            yonghu_cut.extend(words)
        data_yonghu = self.changewords(set(yonghu_cut))
        return data_yonghu

    def loaddicts(self):
        with open(os.path.join(BASE_DIR, 'data/wordsdic.pkl'), 'rb') as f2:
            loaddic = pickle.load(f2)
        return loaddic

    # def emotion(self, strcontent):
    #     classifier = fasttext.load_model(self.model, label_prefix='__label__')
    #     labels_proba = classifier.predict_proba(strcontent)
    #     self.configs['score'] = labels_proba[0][0][1]
    #
    #     if labels_proba[0][0][1] < 0.7:
    #         self.configs['emotion_polarity'] = 'neu'
    #     else:
    #         if labels_proba[0][0][0] == 'neg':
    #             self.configs['emotion_polarity'] = 'neu'
    #         else:
    #             self.configs['emotion_polarity'] = labels_proba[0][0][0]
    #
    #     return self.configs

    def search_words(self, content):
        neg_words = []
        pos_words = []
        neu_words = []
        political_words = []
        school_words = []
        for word in content:
            if self.loaddic.__contains__(word):
                # neg_words.append(word)
                if self.loaddic[word] == 0:
                    school_words.append(word)
                elif self.loaddic[word] == 1:
                    political_words.append(word)
                elif self.loaddic[word] == 2:
                    pos_words.append(word)
                elif self.loaddic[word] == 3:
                    neg_words.append(word)
        self.configs['active_words'] = ','.join(set(pos_words))
        self.configs['negative_words'] = ','.join(set(neg_words))
        self.configs['neutral_words'] = ','.join(set(neu_words))
        self.configs['political_words'] = ','.join(set(political_words))
        self.configs['school_words'] = ','.join(set(school_words))
        return self.configs

    def writeLogger(self, md5):
        with open(os.path.join(BASE_DIR, 'result/logfile.txt'), 'a+') as f:
            f.write('contentmd5:' + str(md5))
            f.write('\n')
        return

    def getscore(self, data, load_vec_xuexiao, p0num_xuexiao):
        if len(data) == 0:
            score = 0
            words_info = ' '
            return score, words_info
        else:
            words_info = {}
            try:

                vec_xuexiao = load_vec_xuexiao.transform(data).toarray()

                p0_words_xuexiao = load_vec_xuexiao.inverse_transform(vec_xuexiao)
                words_info = {"p0_words_xuexiao": list(p0_words_xuexiao[0])}

            except Exception as e:
                print("vec error")
            sum_weight_xuexiao = [0]

            try:
                sum_weight_xuexiao = (np.dot(vec_xuexiao, p0num_xuexiao))

            except Exception as e:
                print("dot error")
            # print sum_weight

            return sum_weight_xuexiao, words_info

    def sumscore(self, weibo_content, weibo_yonghu, weibo_huati):
        outst_content, words_content = self.cutword(weibo_content)
        outst_yonghu, words_yonghu = self.cutword(" ".join(weibo_yonghu))
        outst_huati, words_huati = self.cutword(" ".join(weibo_huati))
        print(outst_huati)
        data_content = self.changewords(words_content)
        data_huati = self.gethuati(words_huati)
        data_yonghu = self.getyonghu(words_yonghu)
        dataset = [data_content, data_huati, data_yonghu]
        score_content, info_content = self.getscore(dataset[0], self.xuexiao_count, self.xuexiao_pv)
        score_huati, info_huati = self.getscore(dataset[1], self.xuexiao_count, self.xuexiao_pv)
        score_yonghu, info_yonghu = self.getscore(dataset[2], self.xuexiao_count, self.xuexiao_pv)
        score = score_content + 1.5 * score_huati + 1.2 * score_yonghu
        word_info = {'content': info_content['p0_words_xuexiao'], 'huati': info_huati['p0_words_xuexiao'],
                     'yonghu': info_yonghu['p0_words_xuexiao']}
        self.configs['related_school'] = score[0]
        self.configs['remarks'] = word_info
        return self.configs

    def analysis(self, contents):
        # stdout_backup = sys.stdout
        # log_file = open(os.path.join(BASE_DIR, 'result/logfile.txt'), 'a+')
        # sys.stdout = log_file
        flag = 0
        result_info = self.configs
        try:

            if contents:
                clist = []
                weibo_content, weibo_yonghu, weibo_huati = self.delherf(contents)
                outstr, words = self.cutword(weibo_content)
                self.search_words(words)
                clist.append(outstr + '\n')
                # self.emotion(clist)
                ##=============================
                # sentiment analysis
                # self.emotion(contents)

                result_info = self.sumscore(weibo_content, weibo_yonghu, weibo_huati)
            else:
                print('no data')
                flag = 1
        except:
            traceback.print_exc()
            print('error')
            flag = 1
        if flag == 1:
            print('error contents is: ' + contents)
        # log_file.close()
        return result_info
