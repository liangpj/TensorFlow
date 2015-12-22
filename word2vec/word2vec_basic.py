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
    for word, _ in count :
        dictionary[word] = len(dictionary)

    data = list()  #使用word ID 来表示原来文本中的数据，并且保持原来数据的有序性
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

del words  # 将少内存
print ('Most commom words （+UNK）', count[:10])
print ("Sample data", data[:10])


data_index = 0


# 生成训练skip-gram 模型的training batch
def generate_batch(batch_size, num_skips, skip_window) :
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1 # [skip_window target skip_window]
    buffer = collections.deque(maxlen=span)
    for _ in range(span) :
        buffer.append((data[data_index]))
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips) :
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips) :
            while target in targets_to_avoid :
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8) :
    print (batch[i], '->', labels[i, 0])
    print (reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])
