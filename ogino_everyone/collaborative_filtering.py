#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

def make_data_small(data):
    #後ろの1週間を除いて100ユーザー分のデータを取得
    former = data[data['time_stamp'] < '2017-04-24 00:00:00.000']
    users_small = former['user_id'].unique()[0:100]
    data_small = former[np.in1d(former['user_id'], users_small)]
    products_small = former['product_id'].unique()
    
    #後ろの1週間から同じユーザーとプロダクトのものを評価用に取得
    latter = data[data['time_stamp'] >= '2017-04-24 00:00:00.000']
    latter = latter[np.in1d(latter['user_id'], users_small)]
    latter = latter[np.in1d(latter['product_id'], products_small)]
    
    test_small = pd.DataFrame(latter['user_id'].unique())
    test_small.columns = ['user_id']
    test_small_ans = latter
    
    return data_small, test_small, test_small_ans

def make_eventcountmat(data, event_type, users = [], products = []):
    if len(users) == 0:
        users = data['user_id'].unique()
    if len(products) == 0:
        products = data['product_id'].unique()
    
    #ユーザーとプロダクトを行列のインデックスに変換
    users_ind = pd.Index(users)
    products_ind = pd.Index(products)
    data['user_id_int'] = data['user_id'].map(lambda x: users_ind.get_loc(x))
    data['product_id_int'] = data['product_id'].map(lambda x: products_ind.get_loc(x))
    
    mat = np.zeros((len(users), len(products)), dtype='float16')
    def count_event(event):
        mat[event['user_id_int'], event['product_id_int']] += 1
        return 0
    data = data[data['event_type'] == event_type]
    data.apply(count_event, axis=1)
    return mat

def make_eventcountmats(data, users = [], products = []):
    if len(users) == 0:
        users = data['user_id'].unique()
    if len(products) == 0:
        products = data['product_id'].unique()
    #各イベントごとにカウント
    mats = np.zeros((4, len(users), len(products)), dtype='float16')
    for i in [0, 1, 2, 3]:
        mats[i, :, :] = make_eventcountmat(data, i, users, products)    
    return mats

def make_crossmat(data = None, users = [], products = [], scores = [], mats = []):
    #スコアの重みをかけて足す
    if len(scores) == 0:
        scores = np.array([
            3, #0カート
            1, #1閲覧
            2, #2クリック
            4  #3コンバージェンス
        ])
    if len(mats) == 0:
        mats = make_eventcountmats(data, users, products)
    crossmat = np.einsum('ijk,i', mats, scores)
    return crossmat

def get_rating_error(r, p, q):
    return r - np.dot(p, q)

def get_error(R, P, Q, beta):
    error = 0.0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] == 0:
                continue
            error += pow(get_rating_error(R[i][j], P[:,i], Q[:,j]), 2)
    error += beta/2.0 * (np.linalg.norm(P) + np.linalg.norm(Q))
    return error

def matrix_factorization(R, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    P = np.random.rand(K, len(R))
    Q = np.random.rand(K, len(R[0]))
    t1 = time.time()
    step = 0
    while True:
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] == 0:
                    continue
                err = get_rating_error(R[i][j], P[:, i], Q[:, j])
                for k in xrange(K):
                    P[k][i] += alpha * (2 * err * Q[k][j])
                    Q[k][j] += alpha * (2 * err * P[k][i])
        error = get_error(R, P, Q, beta)
        if step % 100 == 0:
            time_spent = time.time()-t1
            print("step: " + str(step) + " error: " + str(error) + " time: " + str(time_spent) + "秒")
        step += 1
        if error < threshold or step >= steps:
            time_spent = time.time()-t1
            print("step: " + str(step) + " error: " + str(error) + " time: " + str(time_spent) + "秒")
            break
    return P, Q

def make_recommend(test, mat, users, products):
    users_ind = pd.Index(users)
    recommend_df = pd.DataFrame([[],[]]).T
    for user_id in test['user_id']:
        user_int = users_ind.get_loc(user_id)
        scores = mat[user_int,:]
        ranking = np.argsort(scores)
        recommends = []
        for r in ranking:
            product_id = products[r]
            recommends.append(product_id)
            if len(recommends) >= 22:
                break
        k = len(recommends)
        add = pd.DataFrame([[user_id] * k, recommends, range(k)]).T
        recommend_df = pd.concat([recommend_df, add], axis = 0)
    recommend_df.index = range(recommend_df.shape[0])
    return recommend_df

def evaluate(recommend_df, data_ans):
    rels = [0, 1, 3, 7]
    data_ans['rel'] = data_ans['event_type'].map(lambda x: rels[x])
    i = 0
    scores = []
    for user_id in recommend_df[0].unique():
        a = data_ans[data_ans['user_id'] ==user_id]
        r = recommend_df[recommend_df[0] ==user_id]
        
        a_rel = a.sort_values(by = 'rel', ascending = False)
        a_rel.drop_duplicates('product_id')
        a_rel = a_rel['rel']
        l = min(len(a_rel), 22)
        idcg = 0
        for j in xrange(l):
            idcg += a_rel.values[j] / np.log2(j+2)
        #print("idcg:"+str(idcg))
        
        dcg = 0
        for r_e in r.iterrows():
            j = r_e[1][2]
            a_list = a[a['product_id'] == r_e[1][1]]['rel'].sort_values(ascending = False)
            r_e_rel = 0
            if a_list.size > 0:
                dcg += a_list.values[0] / np.log2(j+2)
        #print("dcg:"+str(dcg))
        
        scores.append(dcg / idcg)
        #i += 1
        #if i > 5:
        #    break
    return np.mean(scores)

def nmf_fill0(R, K, steps=5000, beta=0.02, threshold=0.001, random_state=1234):
    isvalue = (R != 0)
    eps = np.finfo(float).eps
    np.random.seed(random_state)
    P = np.random.rand(K, len(R))
    Q = np.random.rand(K, len(R[0]))
    RT = R.T
    t1 = time.time()
    step = 0
    while True:
        PQzero = np.multiply(np.dot(P.T, Q), isvalue)
        
        Qn = np.dot(P, R)
        Qd = np.dot(P, PQzero) + eps
        #Q = np.matrix(np.array(Q) * np.array(Qn) / np.array(Qd))
        Q = Q * Qn / Qd
        
        Pn = np.dot(Q, RT)
        Pd = np.dot(Q, PQzero.T) + eps
        #P = np.matrix(np.array(P) * np.array(Pn) / np.array(Pd))
        P = P * Pn / Pd
        
        error = get_error(R, P, Q, beta)
        if step % 100 == 0:
            time_spent = time.time()-t1
            print("step: " + str(step) + " error: " + str(error) + " time: " + str(time_spent) + "秒")
        step += 1
        if error < threshold or step >= steps:
            time_spent = time.time()-t1
            print("step: " + str(step) + " error: " + str(error) + " time: " + str(time_spent) + "秒")
            break
    return P, Q

if __name__ == '__main__':
    print("make_data_small")
    filename = '../data/train/train_C.tsv'
    train = pd.read_table(filename)
    train_small, test_small, test_small_ans = make_data_small(train)

    users_small = train_small['user_id'].unique()
    users_small.sort()
    products_small = train_small['product_id'].unique()
    products_small.sort()
    print("event: " + str(len(train_small)))
    print("user: " + str(len(users_small)))
    print("product: " + str(len(products_small)))

    print("make_crossmat")
    mats = make_eventcountmats(train_small, users_small, products_small)
    mat = make_crossmat(mats = mats)

    print("NMF")
    #nP, nQ = matrix_factorization(mat, 5, threshold=1.0)
    #mat_estimate = np.dot(nP.T,nQ)
    from sklearn.decomposition import NMF
    model = NMF(n_components=5, init='random', random_state=9, solver='mu', max_iter=5000)
    P = model.fit_transform(mat)
    Q = model.components_
    mat_estimate = np.dot(P, Q)

    print("make_recommend")
    mat_add = (mats[3] != 0) * (-10)
    submit_df = make_recommend(test_small, mat_estimate + mat_add, users_small, products_small)

    print("evaluate")
    v = evaluate(submit_df, test_small_ans)

    print(v)

