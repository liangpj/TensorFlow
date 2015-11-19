#-*-coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import numpy as np
import os
import random
from six.moves import urllib
from six.moves import xrange # pylint: disable = redefined
import tensorflow as tf
import zipfile




# 步骤1：下载数据集

url = 'http://mattmahoney.net/dc/' # 数据集地址

def maybe_download(filename, expected_bytes) :
    "如果数据集不存在当前目录，则需要下载"
    if not os.path.exists(filename) :
        filename, _ = urllib.request.urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes :
        print("Found and verified", filename)
    else :
        print(statinfo.st_size)
        raise  Exception(
            'Faled to verify ' + filename + '. Can you get to it with a browser ?'
        )
    return filename

filename = maybe_download('text8.zip', 31344016)

# 读取数据
def read_data(filename) :
    f = zipfile.ZipFile(filename)
    for name in f.namelist() :
        return f.read(name).split()
    f.close()

words = read_data(filename)
print("Data size", len(words))


## 步骤二： 建立词汇表和使用UNK取代稀有词

vocabulary_size = 50000

def build_dataset(words) :
    count = [["UNK", -1]]  #创建一个hash表
    count.extend(collections.Counter(words).most_common(vocabulary_size -1)) #只取前面vocabulary个单词

    ## 创建一个词汇表 dictionary = (word, id)
    dictionary = dict()
    for word, _ in words :
        dictionary[words] = len(dictionary)

    data = list()  # word id
    unk_count = 0
    for word in words :
        if word in dictionary :
            index = dictionary[word]
        else :
            index = 0 #dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    # reverse_dictionary = (id, word)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
