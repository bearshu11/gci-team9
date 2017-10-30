#!usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import time
from scipy.sparse import lil_matrix

#疎行列の評価値表からレコメンドを計算する関数
def sparse_collaborative_filtering(scoremat_sparse, test, user_ids, product_ids):
  start = time.time()
  recommend_df = pd.DataFrame([[],[]]).T
  test_user_ids = test['user_id_int'].as_matrix()
  for i in range(len(test_user_ids)):
    #ログ
    if i % 100 == 0:
      print(str(i)+'/'+str(len(test_user_ids))),
      print(time.time() - start),
      print('seconds')
    target_int = test_user_ids[i]
    #対象ユーザーの評価値リスト
    target_scores = scoremat_sparse[target_int]
    #各ユーザーとの類似度(評価値リストの内積)
    corr = scoremat_sparse.dot(target_scores.T)
    #類似度をかけた評価値リストの和
    recommend_scores = corr.T.dot(scoremat_sparse)
    #自分のスコアとの差分を計算
    recommend_scores -= target_scores * corr.sum()
    #ソートしてレコメンド
    ranking = np.argsort(recommend_scores.todense().tolist()[0])
    recommends = []
    for r in ranking:
      product_id = product_ids[r]
      recommends.append(product_id)
      if len(recommends) >= 22:
        break
    k = len(recommends)
    user_id = user_ids[target_int]
    add = pd.DataFrame([[user_id] * k, recommends, range(k)]).T
    recommend_df = pd.concat([recommend_df, add], axis = 0)
  i = len(test_user_ids)
  print(str(i)+'/'+str(len(test_user_ids))),
  print(time.time() - start),
  print('seconds')
  return recommend_df


if __name__ == '__main__':

  print('train読み込み')
  train = pd.read_table('../../data/train/train_A.tsv')
  #スコア行列の行に対応するユーザーIDのリスト
  user_ids = train['user_id'].unique()
  user_ids.sort()
  #スコア行列の列に対応する商品IDのリスト
  product_ids = train['product_id'].unique()
  product_ids.sort()

  print('test読み込み')
  test = pd.read_table('../../test.tsv')
  #テストデータからユーザーidのカテゴリがAのものを取得
  test_C = test[test['user_id'].map(lambda x: x[-1]) == 'A'].copy()

  print('ID -> 数字')
  user_ids_index = pd.Index(user_ids)
  product_ids_index = pd.Index(product_ids)
  train['user_id_int'] = train['user_id'].map(lambda x : user_ids_index.get_loc(x))
  train['product_id_int'] = train['product_id'].map(lambda x : product_ids_index.get_loc(x))
  test_C['user_id_int'] = test_C['user_id'].map(lambda x : user_ids_index.get_loc(x))
  print('クラスター読込')
  user_cluster = pd.read_csv('user_cluster_A.csv')
#   product_cluster = pd.read_csv('product_cluster.csv')

  user_cluster.columns = ['user_id', 'user_cluster']
  train = train.merge(user_cluster, on='user_id')
#   product_cluster.columns = ['product_id_int', 'product_cluster']
#   train = train.merge(product_cluster, on='product_id_int')

  print('評価値計算')
  start = time.time()
  #評価値行列(疎行列のscipy.sparse.lil_matrix型。numpy.matrixと大体同じように使える)
  #小数を使う場合はdtypeを変えて下さい
  scoremat = lil_matrix((len(user_ids), len(product_ids)), dtype=np.float16)
#   scores = [1.0, 1.0, 3.0, 7.0]
  ca_values = {0:0.0001, 1:0.03, 2:0.0001, 3:0.08, 4:0.56, 5:0.002, 6:0, 7:0.35, 8:0.006}
  pd_values = {0:0.0001, 1:0.0001, 2:0.0001, 3:0.18, 4:0.31, 5:0.003, 6:0, 7:0.30, 8:0.007}
  cl_values = {0:0.002, 1:0.01, 2:0.0001, 3:0.09, 4:0.08, 5:0.0001, 6:0, 7:0.11, 8:0.003}
  cv_values = {0:0.02, 1:0.13, 2:0.2, 3:0.35, 4:2.34, 5:0.01, 6:0, 7:1.00, 8:0.04}
  
  for event in train[['user_id_int', 'product_id_int', 'event_type','user_cluster']].as_matrix():
    i = event[0]
    j = event[1]
    event_type = event[2]
    cluster = event[3]
    value = 0
    
    if event_type == 0:
        value += ca_values[cluster]
    elif event_type == 1:
        value += pd_values[cluster]
    elif event_type == 2:
        value += cl_values[cluster]
    elif event_type == 3:
        value += cv_values[cluster]
    
    scoremat[i,j] += scores[e]

  #正規化
  L = scoremat.multiply(scoremat).sum(axis=1)
  print(np.where(L==0))
  L[L == 0] = 1
  L = np.power(L, -0.5)
  scoremat = scoremat.multiply(L).asformat('lil')

  print(time.time() - start),
  print('seconds')

  print('レコメンドを計算')
  submit_df = sparse_collaborative_filtering(scoremat, test_C, user_ids, product_ids)

  print('提出形式で保存')
  submit_df.to_csv('submit_A.tsv', sep='\t', header=False, index=False, encoding='utf-8', line_terminator='\r\n')

