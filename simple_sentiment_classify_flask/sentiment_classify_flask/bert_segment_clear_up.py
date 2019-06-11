#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    :  bert_segment_clear_up.py
# @Date    :  2019/5/8 14:38
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
import re
from collections import Iterable
from lxml import etree

from .settings import _LOGGINE_NAME

logger = logging.getLogger(_LOGGINE_NAME)


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
