import numpy as np
from datetime import datetime as dt

class DataProcessor():

    def __init__(self, data):
        self.data = data

    def id_str2int(self, str_id):
        return int(str_id.split("_")[0])

    def compute_value(self, actions):
        """
        評価値を算出するメソッド

        Params
        -------
        actions: list
        ユーザーの一つのプロダクトに関する行動のリスト

        actions[i][0]: int "event_type"
        actions[i][1]: int "ad"
        actions[i][2]: datetime "time_stamp"

        注意：i は 0<i<len(actions) の整数。
        """
        # TODO:PLEASE FILL ME!
        value = 1
        return value

    def get_max_ids(self, plusone=True):
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
        cf_data=dict()
        for row in self.data:
            user_id = self.id_str2int(row[0])
            product_id = self.id_str2int(row[1])
            if not user_id in cf_data.keys():
                cf_data[user_id]=dict()
            if not product_id in cf_data[user_id].keys():
                cf_data[user_id][product_id]=[]
            action = [int(row[2]),int(row[3]),dt.strptime(row[4].split(".")[0], '%Y-%m-%d %H:%M:%S')]
            cf_data[user_id][product_id].append(action)

        cf_dict=dict()
        for user_id, user_actions in cf_data.items():
            cf_dict[user_id] = dict()
            for product_id, actions in user_actions.items():
                value = self.compute_value(actions)
                cf_dict[user_id][product_id] = value

        return cf_dict

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
