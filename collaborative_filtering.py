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

def make_crossmat(data, users = None, products = None):
    if users == None:
        users = data['user_id'].unique()
    if products == None:
        products = data['product_id'].unique()
    
    #ユーザーとプロダクトを行列のインデックスに変換
    data['user_id_int'] = data['user_id'].map(lambda x: np.where(users == x)[0][0])
    data['product_id_int'] = data['product_id'].map(lambda x: np.where(products == x)[0][0])
    
    #各イベントごとにカウント
    mats = np.zeros((4, len(users), len(products)))
    def count_event(event):
        mats[event['event_type'], event['user_id_int'], event['product_id_int']] += 1
        return 0
    train_small.apply(count_event, axis=1)
    
    #スコアの重みをかけて足す
    scores = np.array([
        3, #0カート
        1, #1閲覧
        2, #2クリック
        4  #3コンバージェンス
    ])
    crossmat = np.einsum('ijk,i', mats, scores)
    
    return mats, crossmat

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
            print("step: " + str(step) + " error: " + str(error) + " time: " + str(time_spent) + "秒\r"),
        step += 1
        if error < threshold or step >= steps:
            time_spent = time.time()-t1
            print("step: " + str(step) + " error: " + str(error) + " time: " + str(time_spent) + "秒\r"),
            break
    return P, Q

def make_recommend(test, mat, exclude_mat, users, products):
    recommend_df = pd.DataFrame([[],[]]).T
    for user_id in test['user_id']:
        user_int = np.where(users == user_id)[0][0]
        scores = mat[user_int,:]
        ranking = np.argsort(scores)
        recommends = []
        for r in ranking:
            if not exclude_mat[user_int,r]:
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

print("make_data_small")
filename = 'data/train/train_C.tsv'
train = pd.read_table(filename)
train_small, test_small, test_small_ans = make_data_small(train)

users_small = train_small['user_id'].unique()
products_small = train_small['product_id'].unique()
print("event: " + str(len(train_small)))
print("user: " + str(len(users_small)))
print("product: " + str(len(products_small)))

print("make_crossmat")
mats, mat = make_crossmat(train_small, users_small, products_small)

print("matrix_factorization")
nP, nQ = matrix_factorization(mat, 5, threshold=1.0)
mat_estimate = np.dot(nP.T,nQ)

print("make_recommend")
exclude_mat = (mats[3] != 0)
submit_df = make_recommend(test_small, mat, exclude_mat, users_small, products_small)

print("evaluate")
v = evaluate(submit_df, test_small_ans)

print(v)

