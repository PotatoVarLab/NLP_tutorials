#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    :   sent_dict.py
# @Date    :   2019/06/11 14:54:31
# @Author  :   H.wei
# @Version :   1.0
# @License :   Copyright (C) 2019 YPLSEC, Inc.

# @Desc    :   None

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations


class SentDict(object):
    def __init__(self, docs=[], method="PMI", min_times=5, ft_size=100, ft_epochs=15, ft_window=5, pos_seeds=[], neg_seeds=[]):
        super(SentDict, self).__init__()
        self.build_sent_dict(docs, method, min_times, ft_size,
                             ft_epochs, ft_window, pos_seeds, neg_seeds)

    def __getitem__(self, key):
        return self.sent_dict[key]

    def set_pos_seeds(self, pos_seeds):
        self.pos_seeds = [w for w in pos_seeds if w in self.words]

    def set_neg_seed(self, neg_seeds):
        self.neg_seeds = [w for w in neg_seeds if w in self.words]

    def build_sent_dict(self, docs=[], method="PMI", min_times=5, ft_size=100, ft_epochs=15, ft_window=5, pos_seeds=[], neg_seeds=[]):
        self.doc_count = len(docs)
        self.method = method
        if self.doc_count > 0:
            if method == "PMI":
                self.co_occur, self.one_occur = self.get_word_stat(docs)
                self.words = [
                    word for word in self.one_occur if self.one_occur[word] >= min_times]
                # 如果有新的输入，就更新种子词，否则默认已有（比如通过set已设定）
                if len(pos_seeds) > 0 and len(neg_seeds) > 0:
                    self.pos_seeds = [w for w in pos_seeds if w in self.words]
                    self.neg_seeds = [w for w in neg_seeds if w in self.words]
                if len(self.pos_seeds) > 0 and len(self.neg_seeds) > 0:
                    self.sent_dict = self.SO_PMI(self.words)
            elif method == "fasttext":
                from gensim.models import FastText
                self.fasttext = FastText(
                    docs, iter=ft_epochs, size=ft_size, window=ft_window, min_count=min_times)
                self.words = self.fasttext.wv.index2word
                self.pos_seeds = [w for w in pos_seeds if w in self.words]
                self.neg_seeds = [w for w in neg_seeds if w in self.words]
                if len(self.pos_seeds) > 0 and len(self.neg_seeds) > 0:
                    self.sent_dict = self.SO_FastText(self.words)

    def analyse_sent(self, words):
        if self.method == "PMI":
            words = [word for word in words if word in self.sent_dict]
            if len(words) > 0:
                return sum(self.sent_dict[word] for word in words) / len(words)
            else:
                return 0
        elif self.method == "fasttext":
            cnt = 0
            senti = 0.0
            for word in words:
                if word in self.sent_dict:
                    cnt += 1
                    senti += self.sent_dict[word]
                else:
                    try:
                        SO_ft0 = sum(self.FastTextSim(word, seed) for seed in self.pos_seeds) - \
                            sum(self.FastTextSim(word, seed)
                                for seed in self.neg_seeds)
                        cnt += 1
                        senti += SO_ft0
                    except:
                        continue
            if cnt == 0:
                return 0.0
            return (senti/cnt)

    def get_word_stat(self, docs, co=True):
        co_occur = dict()               # 由于defaultdict太占内存，还是使用dict
        one_occur = dict()
        for doc in docs:
            for word in doc:
                if not word in one_occur:
                    one_occur[word] = 1
                else:
                    one_occur[word] += 1
            if co:
                for a, b in combinations(doc, 2):
                    if not (a, b) in co_occur:
                        co_occur[(a, b)] = 1
                        co_occur[(b, a)] = 1
                    else:
                        co_occur[(a, b)] += 1
                        co_occur[(b, a)] += 1
        print('get_word_stat', '-->', 'one_occur: ', one_occur)
        print('get_word_stat', '-->', 'co_occur: ', co_occur)
        return co_occur, one_occur

    def PMI(self, w1, w2):
        if not((w1 in self.one_occur) and (w2 in self.one_occur)):
            raise Exception()
        if not (w1, w2) in self.co_occur:
            return 0
        c1, c2 = self.one_occur[w1], self.one_occur[w2]
        c3 = self.co_occur[(w1, w2)]
        return np.log2((c3*self.doc_count)/(c1*c2))

    def SO_PMI(self, words):
        ret = {}
        for word in words:
            ret[word] = sum(self.PMI(word, seed) for seed in self.pos_seeds) - \
                sum(self.PMI(word, seed) for seed in self.neg_seeds)
        return ret

    def FastTextSim(self, w1, w2):
        try:
            vec1, vec2 = self.fasttext.wv[w1], self.fasttext.wv[w2]
        except:
            return 0.0
        return np.dot(vec1, vec2)/np.sqrt(np.dot(vec1, vec1)*np.dot(vec2, vec2))

    def SO_FastText(self, words):
        ret = {}
        for word in words:
            ret[word] = sum(self.FastTextSim(word, seed) for seed in self.pos_seeds) - \
                sum(self.FastTextSim(word, seed) for seed in self.neg_seeds)
        return ret


if __name__ == "__main__":
    
    docs = [["武磊", "威武", "，", "中超", "第一", "射手", "太", "棒", "了", "！"],
            ["武磊", "强", "，", "中超", "最", "棒", "球员"],
            ["郜林", "不行", "，", "只会", "抱怨", "的", "球员", "注定", "上限", "了"],
            ["郜林", "看来", "不行", "，", "已经", "到", "上限", "了"]]

    def read_file(fn):
        words = []
        with open(fn, 'r') as fr:
            for line in fr:
                if line.strip():
                    words.append(line.strip().lower())
        return words

    def read_gbk_file(fn):
        words = []
        with open(fn, 'rb') as fr:
            for line in fr:
                if line.decode().strip():
                    words.append(line.decode().strip().lower())
        return words

    def write_words(fn, words):
        with open(fn, 'w', encoding='utf-8') as fw:
            for w in words:
                fw.write(w + '\n')

    def r_fn(fn):
        words = []
        with open(fn, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line.strip():
                    words.append(line.strip().lower())
        return words


    # pos_fn1 = 'net_positives.txt'
    # pos_fn2 = 'ntusd-positive.txt'
    # pos_fn3 = 'tsinghua.positive.gb.txt'
    # pos_words = read_file(pos_fn1) + read_gbk_file(pos_fn2) + read_file(pos_fn3)
    # # pos_words = read_gbk_file(pos_fn2)
    # write_words('pos_words.txt', pos_words)

    # neg_fn1 = 'net_negatives.txt'
    # neg_fn2 = 'ntusd-negative.txt'
    # neg_fn3 = 'tsinghua.negative.gb.txt'
    # neg_words = read_file(neg_fn1) + read_gbk_file(neg_fn2) + read_file(neg_fn3)
    # # neg_words = read_gbk_file(neg_fn2)
    # print(len(neg_words))
    # print(neg_words[:10])
    # write_words('neg_words.txt', neg_words)
    # exit(0)

    neg_words = r_fn('neg_words.txt')
    print('negative words len: {}'.format(len(neg_words)))
    pos_words = r_fn('pos_words.txt')
    print('positive words len: {}'.format(len(pos_words)))
    # exit(0)

    # sent_dict = SentDict(docs, method="PMI", min_times=1,
    #                      pos_seeds=pos_words, neg_seeds=neg_words)
    # print("威武", sent_dict["威武"])
    # print("球员", sent_dict["球员"])
    # print("上限", sent_dict["上限"])
    # print(sent_dict.analyse_sent(docs[0]))

    # sent_dict = SentDict(docs, method="fasttext", min_times=1,
    #                      pos_seeds=pos_words, neg_seeds=neg_words)
    # print("威武", sent_dict["威武"])
    # print("球员", sent_dict["球员"])
    # print("上限", sent_dict["上限"])
    # print(sent_dict.analyse_sent(docs[0]))
