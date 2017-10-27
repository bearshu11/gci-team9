#!usr/bin/python
# -*- coding:utf-8 -*-


#import main_tsuchihashi
#mat = main_tsuchihashi.main_tsu()

import numpy as np
import pandas as pd
import time
import sys

print(time.ctime())
print('START')

sys.path.append('../ogino_everyone')

import collaborative_filtering as cf

train = pd.read_table('../data/train/train_C.tsv')

#スコア行列の行に対応するユーザーIDのリスト
user_ids = train['user_id'].unique()
user_ids.sort()
#スコア行列の列に対応する商品IDのリスト
product_ids = train['product_id'].unique()
product_ids.sort()
#NMF後のスコア行列(ここでは例として乱数行列を用いてます)
#mat = np.random.rand(len(user_ids), len(product_ids)).astype('float16')

print(time.ctime())
print('START eventcount')

# TODO hyoukachimat
mats = cf.make_eventcountmats(train, user_ids, product_ids)
mat = mats[3] * 5
mat2 = mats[1] * 0.3 + mats[2] * 0.3
mat2[mat2 > 4.5] = 4.5
mat += mat2
del mat2

print(time.ctime())
print('user_ids')
print(user_ids)
print('product_ids')
print(product_ids)
print('mat')
print(mat)


from sklearn.decomposition import NMF

from numpy import nan as NA
#穴あき評価値行列
#scoremat = np.array([
#    [1, 2, 3, NA, NA],
#    [4, 5, NA, 6, NA],
#    [NA, 7, 8, 9, 1],
#    [2, NA, 3, 4, 5]
#])

scoremat[scoremat == 0] = NA


#n_componentsはユーザーの特徴ベクトルの数なので、本番では100~1000くらいでしょうか?
model = NMF(n_components=100, init='random', random_state=1234, solver='mu', max_iter=200)

P = model.fit_transform(scoremat)
Q = model.components_
mat = np.dot(P, Q)

print(time.ctime())
print('scoremat')
print(scoremat)
print('mat')
print(mat)



#mat_add = ((cf.make_eventcountmat(data=train, event_type=3, users=user_ids, products=product_ids) != 0) * (-10)).astype('float16')
mat_add = (mats[3] != 0) * (-10)

test = pd.read_table('../data/test.tsv')

#テストデータからユーザーidのカテゴリがCのものを取得
test_C = test[test['user_id'].map(lambda x: x[-1]) == 'C']
print(time.ctime())
print('test_C')
test_C.head()


#テストデータ、スコア行列、ユーザーIDリスト、商品IDリストからレコメンドを求める
submit_C = cf.make_recommend(test_C, mat + mat_add, user_ids, product_ids)
print(time.ctime())
print('submit_C')
submit_C.head()

#提出形式で保存
submit_C.to_csv('submit_C.tsv', sep='\t', header=False, index=False, encoding='utf-8', line_terminator='\r\n')

print(time.ctime())

