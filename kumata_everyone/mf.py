import numpy as np
from datetime import datetime as dt

class MatrixFactorization():
    def get_rating_error(self, r, p, q):
        s = 0
        for i in range(len(p)):
            s += p[i]*q[i]
        return r - s

    def get_error(self, R, P, Q, beta, sorted_user_ids, sorted_product_ids):
        error = 0.0
        for i, user_id in enumerate(sorted_user_ids):
            exist_product_ids = R[user_id].keys()
            for product_id in exist_product_ids:
                j = sorted_product_ids.index(product_id)
                error += pow(self.get_rating_error(R[user_id][product_id], P[i], Q[j]), 2)
        error += beta/2.0 * (np.linalg.norm(P) + np.linalg.norm(Q))
        return error


    def factorize(self, R, sorted_user_ids, sorted_product_ids, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
        n_user = len(sorted_user_ids)
        n_product = len(sorted_product_ids)
        P = np.random.rand(n_user,K).tolist()
        Q = np.random.rand(n_product,K).tolist()

        for step in range(steps):
            for i, user_id in enumerate(sorted_user_ids):
                exist_product_ids = R[user_id].keys()
                for product_id in exist_product_ids:
                    j = sorted_product_ids.index(product_id)
                    err = self.get_rating_error(R[user_id][product_id], P[i], Q[j])
                    for k in range(K):
                        P[i][k] += alpha * (2 * err * Q[j][k])
                        Q[j][k] += alpha * (2 * err * P[i][k])
            error = self.get_error(R, P, Q, beta)
            if error < threshold:
                print("FINISHED! step is " + str(step))
                break
        return P, Q

        # for step in range(steps):
        #     for i in range(n_user):
        #         exist_product_ids = R[i].keys()
        #         for j in exist_product_ids:
        #             err = self.get_rating_error(R[i][j], P[i], Q[j])
        #             for k in range(K):
        #                 P[i][k] += alpha * (2 * err * Q[j][k])
        #                 Q[j][k] += alpha * (2 * err * P[i][k])
        #     error = self.get_error(R, P, Q, beta)
        #     if error < threshold:
        #         print("FINISHED! step is " + str(step))
        #         break
        # return P, Q