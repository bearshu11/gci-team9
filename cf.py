import numpy as np
from datetime import datetime as dt
import sys

class DataProcessor():

    def __init__(self, data):
        """
        イニシャライザ
        Params
        ------
        data: list
        データの読み込み方は、main.py参照
        """
        self.data = data
        print(self.data[0])
        sys.exit()

    def id_str2int(self, str_id):
        """
        idを文字からint型に変換する。
        idをint型で扱った方が、評価値の表にするときにidをindexとすることができ、扱いやすいため。
        Example
        ------
        "000000_A" => 0
        "000034_A" => 34
        "000103_a" => 103
        """
        return int(str_id.split("_")[0])

    def compute_value(self, actions):
        """
        評価値を算出するメソッド
        Params
        -------
        actions: list
        ユーザーのある一つのプロダクトに関する行動のリスト
        actions[i][0]: int "event_type"
        actions[i][1]: int "ad"
        actions[i][2]: datetime "time_stamp"
        注意：i は 0<i<len(actions) の整数。
        """
        # TODO:カテゴリーごとに自由に設定する。
        # TODO:カテゴリーごとに定めたクラスターを有効利用できれば、なおよい。
        value = 1
        return value

    def get_user_cluster(self, user_id):
        """
        ユーザーのクラスター番号を返すメソッド
        Params
        -------
        user_id: int
        """
        # TODO:intで返せればintで返したい
        user_cluster = 1
        return user_cluster
    
    def get_product_cluster(self, product_id):
        """
        プロダクトのクラスター番号を返すメソッド
        Params
        -------
        user_id: int
        """
        # TODO:intで返せればintで返したい
        product_cluster = 1
        return product_cluster

    def get_max_ids(self, plusone=True):
        """
        最大のuser_idとproduct_idを得るメソッド
        (ユニークなuser_idとproduct_idの数を求めるために使う)
        Parameter
        -----
        plusone: Boolean
        求めた最大のidに1を足すかどうか決める引数。
        ユニークなidの数を数えるとき、idが0から始まっていると、idの最大値だけでは1足りないため。
        """
        max_user_id = 0
        max_product_id = 0
        for row in self.data:
            user_id = self.id_str2int(row[0])
            product_id = self.id_str2int(row[1])
            if max_user_id < user_id:
                max_user_id = user_id
            if max_product_id < product_id:
                max_product_id = product_id
        if plusone:
            max_user_id += 1
            max_product_id += 1

        return [max_user_id, max_product_id]

    def make_matrix_for_CF(self):
        """
        MFを適用できる表(dict型)を作成する。（評価値の表を作る）
        cf_dataの形式：
        {user_id: {product_id: [[event_type, ad, time_stamp], ...], ...}, ...}
        cf_dataのイメージ：
        {0: {9250: [[1 , -1, 2017-04-29 00:26:20], [3 , 1, 2017-04-30 00:30:56]],
            14068: [[0 , -1, 2017-04-29 01:26:20]],
             ...},
        34: {394: [[1 , -1, 2017-04-15 06:26:20], [0 , 1, 2017-04-16 00:35:56]],
             ...},
        ...}
        cf_dictの形式：
        {user_id: {product_id: 評価値, ...}, ...}
        cf_dictのイメージ：
        {0: {9250: 2,
             14068: 1,
             1254: 1,
             3316: 1,
             3009: 1,
             4433: 2,
             8525: 1,
             9753: 2},
        34: {394: 2,
            1555: 1,
            10093: 1},
        108: {39: 2,
              ...},
        ...}
        """

        #
        cf_data=dict()
        for row in self.data:
            user_id = self.id_str2int(row[0])
            product_id = self.id_str2int(row[1])
            if not user_id in cf_data.keys():
                cf_data[user_id]=dict()
            if not product_id in cf_data[user_id].keys():
                cf_data[user_id][product_id]=[]
            # 時刻の小数点以下切り捨て
            action = [int(row[2]),int(row[3]),dt.strptime(row[4].split(".")[0], '%Y-%m-%d %H:%M:%S')]

            cf_data[user_id][product_id].append(action)


        cf_dict=dict()
        for user_id, user_actions in cf_data.items():
            cf_dict[user_id] = dict()
            for product_id, actions in user_actions.items():
                value = self.compute_value(actions)
                cf_dict[user_id][product_id] = value

        return cf_dict

    def make_cluster_matrix(self):

class MatrixFactorization():
    def get_rating_error(self, r, p, q):
        s = 0
        for i in range(len(p)):
            s += p[i]*q[i]
        return r - s

    def get_error(self, R, P, Q, beta):
        error = 0.0
        for i in range(len(R)):
            exist_product_ids = R[i].keys()
            for j in exist_product_ids:
                error += pow(self.get_rating_error(R[i][j], P[i], Q[j]), 2)
        error += beta/2.0 * (np.linalg.norm(P) + np.linalg.norm(Q))
        return error

    def factorize(self, R, matrix_size, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
        n_user = matrix_size[0]
        n_product = matrix_size[1]
        P = np.random.rand(n_user,K).tolist()
        Q = np.random.rand(n_product,K).tolist()

        for step in range(steps):
            for i in range(n_user):
                exist_product_ids = R[i].keys()
                for j in exist_product_ids:
                    err = self.get_rating_error(R[i][j], P[i], Q[j])
                    for k in range(K):
                        P[i][k] += alpha * (2 * err * Q[j][k])
                        Q[j][k] += alpha * (2 * err * P[i][k])
            error = self.get_error(R, P, Q, beta)
            if error < threshold:
                print("FINISHED! step is " + str(step))
                break
        return P, Q