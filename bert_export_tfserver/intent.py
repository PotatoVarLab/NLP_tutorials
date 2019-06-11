#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from datetime import datetime
# from predict import predicts
from predict import PredictHandler


name = re.compile(r'@[^,，：:\s@]+')  # find weibo name
emoji_1 = re.compile(r'\[\S+?\]') # find weibo emoji
emoji_2 = re.compile(r'\[\s{0,}\S+?|\]')

def special_char_replace(text):
    ## 文件数据去除多余标点
    s = re.sub(r'，', ',', text)
    s = re.sub(r'。', '.', s)
    s = re.sub(r'？', '?', s)
    s = re.sub(r'！', '!', s)
    
    s = re.sub(r'~{2,}', '~', s)
    s = re.sub(r',{2,}', ',', s)
    s = re.sub(r'\.{2,}', '.', s)
    s = re.sub(r'\${2,}', ';', s)
    s = re.sub(r'-{2,}', '-', s)
    s = re.sub(r'·{2,}', '·', s)
    s = re.sub(r'(\?\s){2,}','?',s)
    s = re.sub(r'\?{2,}', '?', s)
    s = re.sub(r'!{2,}', '!', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s

def remove_nickname(text):
    # 删除emoji和昵称
    names = name.findall(text)
    for n in names:
        text = text.replace(n, ' ')
    text = text.replace('//', '')

    emojis = emoji_1.findall(text)
    for em in emojis:
        text = text.replace(em, ' ')

    emojis = emoji_2.findall(text)
    for em in emojis:
        text = text.replace(em, ' ')
    line = re.sub(r'\s:', '', text)
    line = re.sub(r'\s{2,}', '', line)
    line = re.sub(r'\[', '', line)
    line = re.sub(r':', '', line)
    line = re.sub(r'：', '', line)
    line = re.sub(r'【', ' ', line)
    line = re.sub(r'】', ' ', line)
    return line


pred_handler = PredictHandler()

## singel test
test_content = '这是带头要让人民币升值啊！ //@互联网的那点事:充值面额目前支持 50元，100元，300元 和 500元 ！ 关于定价 $0.99 的应用，中国区定价 ? 6 元。 $1.99 的应用，中国区定价 ? 12 元。 [泪]'
test_content = '谁欺负双鱼座,俺就和他玩儿命!!![抓狂] //@重口味女秘书:双鱼座顶起 ...'
test_content = '杆头打飞了。[泪]'
test_content = '苛捐杂税猛于虎[衰]'

test_content = '''认可996的话，我们的社会就可能步日本后尘，高工作压力、低欲望、高自杀率、低生育率'''


sentence = special_char_replace(remove_nickname(test_content))
st = datetime.now()
dic = pred_handler.predicts([sentence])
print(' \nContent: {}'.format(sentence))
print(dic)
print('Use time:{}'.format(datetime.now() - st))


## series test
# sentences = []
# st = datetime.now()
# for sentence in sentences:
#     dic = pred_handler.predicts([sentence])
#     print(' \ndoc text: {}'.format(sentence))
#     print(dic)
#     print(' \nUse time:{}'.format(datetime.now() - st))
