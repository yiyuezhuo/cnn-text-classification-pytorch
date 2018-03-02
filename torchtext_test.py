# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:20:14 2018

@author: yiyuezhuo
"""

import os
#import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets


class Args:
    pass

args = Args()
args.batch_size = 64
kargs = {'device':-1,  # force cpu mode
         'repeat':False} 

text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_data, dev_data = mydatasets.MR.splits(text_field, label_field)

print(len(train_data.examples),len(dev_data.examples))
print(train_data.fields)
example = train_data.examples[0]
print(example.text)
print(example.label)
print(hasattr(text_field,'vocab'))
text_field.build_vocab(train_data, dev_data)
#text_field 依靠train_data/dev_data里那个fields映射。知道每个example对象
#的哪个字段是自己这个field的，如text_field在example里有个text属性，而text属性
#又在之前就被映射到了text_field这个对象。
label_field.build_vocab(train_data, dev_data)
print(hasattr(text_field,'vocab'))
print(list(text_field.vocab.stoi.items())[:10])



train_iter, dev_iter = data.Iterator.splits(
                            (train_data, dev_data), 
                            batch_sizes=(args.batch_size, len(dev_data)),
                            **kargs)
for it in train_iter:
    break
print(it.text)
print(it.label)
print(len(list(train_iter)), len(list(dev_iter)))
print(len(train_data.examples),len(list(train_iter))*64)

