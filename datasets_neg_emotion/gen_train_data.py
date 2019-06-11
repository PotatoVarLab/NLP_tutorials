#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    :   gen_train_data.py
# @Date    :   2019/06/05 11:50:20
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
import re
from collections import Iterable
from lxml import etree

logger = logging.getLogger(__file__)


class BERTClearUpSegmentBase(object):

    def __init__(self):
        self.__name = re.compile(r'@[^,，：:\s@]+')  # find weibo name
        self.__emoji_1 = re.compile(r'\[\S+?\]')  # find weibo emoji
        self.__emoji_2 = re.compile(r'\[\s{0,}\S+?|\]')

    def clear_up_str(self, content):
        # clear up string and format string
        try:
            text = self._lxml_parse_text(content)
            c_text = self._special_char_replace(
                self._remove_nickname_emoji(text))
        except Exception as e:
            logger.exception(e)
            c_text = ''
        return c_text

    def _lxml_parse_text(self, text=''):
        """Use lxml to parse HTML,extract body and some tags texts.
        """
        clear_text = ''
        try:
            html_etree = etree.HTML(text)
        except Exception as e:
            html_etree = None

        if html_etree is not None:
            bodytexts = html_etree.xpath(
                '//body/descendant-or-self::node()/text()')
            clear_text = ' '.join(self._clear_up_list_strings(bodytexts))
        else:
            pass
        return clear_text

    @staticmethod
    def _clear_up_list_strings(textlist):
        """Clear up some strs in the list.
        """
        clear_temp = []
        if isinstance(textlist, Iterable):
            for s in textlist:
                if isinstance(s, str):
                    s = re.sub('\n', ' ', s)
                    s = re.sub('\t', ' ', s)
                    s = re.sub('\s{2,}', '', s)
                    s = re.sub('\xa0', '', s)
                    s = re.sub('\u3000', '', s)
                    s = re.sub('•', '', s)
                    # s = re.sub('\.', '', s)
                    # s = re.sub('_', '', s)
                    # s = re.sub('＿', '', s)
                    s = re.sub('\|', ' ', s)
                    if s.strip():
                        clear_temp.append(s.lower().strip())
        return clear_temp

    @staticmethod
    def _special_char_replace(text):
        # 文件数据去除多余标点
        # s = re.sub(r'，', ',', text)
        # s = re.sub(r'。', '.', s)
        # s = re.sub(r'？', '?', s)
        # s = re.sub(r'！', '!', s)
        s = re.sub(r'~{2,}', '~', text)
        s = re.sub(r',{2,}', ',', s)
        s = re.sub(r'\.{2,}', '.', s)
        s = re.sub(r'\${2,}', ';', s)
        s = re.sub(r'-{2,}', '-', s)
        s = re.sub(r'·{2,}', '·', s)
        s = re.sub(r'(\?\s){2,}', '?', s)
        s = re.sub(r'\?{2,}', '?', s)
        s = re.sub(r'!{2,}', '!', s)
        s = re.sub(r'\s{2,}', ' ', s)
        s = re.sub(r'\[', '', s)
        return s

    @staticmethod
    def _filter_emoji(desstr, restr=''):
        '''过滤表情emoji
        '''
        try:
            # co = re.compile(u'[\U00010000-\U0010ffff]')
            co = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u200d"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\u3030"
                            u"\ufe0f"
                            "]+", flags=re.UNICODE)
        except re.error:
            co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        return co.sub(restr, desstr)

    def _remove_nickname_emoji(self, text):
        # 删除emoji和昵称
        names = self.__name.findall(text)
        for n in names:
            text = text.replace(n, ' ')
        text = text.replace('//', '')

        emojis = self.__emoji_1.findall(text)
        for em in emojis:
            text = text.replace(em, ' ')

        emojis = self.__emoji_2.findall(text)
        for em in emojis:
            text = text.replace(em, ' ')

        text = self._filter_emoji(text)
        table = {ord(f): ord(t) for f, t in zip(
            '，。“”！？【】（）％＃＠＆１２３４５６７８９０', ',.""!?[]()%#@&1234567890')}
        line = text.translate(table)
        return line


clear_hander = BERTClearUpSegmentBase()


def read_file(fn, label):
    dataset = []
    with open(fn, 'r') as fr:
        for line in fr:
            if line.strip():
                dataset.append(str(label) + '\t' +
                               clear_hander.clear_up_str(line.strip()))
    print('Data lines: {}'.format(len(dataset)))
    return dataset


def gen_train_samples(pos_fn='', neg_fn='', neu_fn=''):
    pos_label = '0'
    neg_label = '1'
    neu_label = '2'

    if pos_fn:
        pos_dataset = read_file(pos_fn, pos_label)
    else:
        pos_dataset = []

    if neg_fn:
        neg_dataset = read_file(neg_fn, neg_label)
    else:
        neg_dataset = []

    if neu_fn:
        neu_dataset = read_file(neu_fn, neu_label)
    else:
        neu_dataset = []

    train_dataset = pos_dataset + neg_dataset + neu_dataset
    train_fn = 'train.tsv' if pos_dataset and neg_dataset and neu_dataset else 'opt.tsv'
    with open(train_fn, 'w+', encoding='utf-8') as fw:
        for line in train_dataset:
            fw.write(line + '\n')

    print('train data over...')


if __name__ == "__main__":
    pos_fn = 'wb_positive_marked.txt'
    neg_fn = 'wb_negative_marked.txt'
    neu_fn = 'wb_neutral_marked_4k.txt'
    gen_train_samples(pos_fn, neg_fn, neu_fn)

    neu_fn = 'wb_neutral_marked_others.txt'
    gen_train_samples(neu_fn=neu_fn)
