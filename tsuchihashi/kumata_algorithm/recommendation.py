# coding:utf-8

import math
import time

class Recommendation():
    """
    評価値のマトリックスからそのまま類似度を計算して、推薦するプロダクトを算出するクラス
    """

    def __init__(self, scores, target_user_ids, user_ids, product_ids):
        """
        Params
        ------
        scores: dict
        評価値のマトリックスをdict型で表したもの

        target_user_ids: list(int)
        推薦するプロダクトを算出したいuser_idのlist

        user_ids: list(int)
        存在するすべてのuser_idのlist

        product_ids: list(int)
        存在するすべてのproduct_idのlist

        """
        self.scores = scores
        self.target_user_ids = target_user_ids
        self.user_ids = user_ids
        self.product_ids = product_ids

    def get_mean_score(self, target_scores):
        """
        特定のユーザーの評価値の平均を算出する。

        Params
        -------
        target_scores: dict(int:int)
        特定のユーザーのプロダクトに対する評価値すべて。

        Returns
        -------
        target_mean_score: float
        ユーザーの各々のプロダクトに対する評価値の平均

        """
        target_total_score = sum(target_scores.values())
        target_mean_score = target_total_score / len(target_scores)

        return target_mean_score


    def get_correlation_coefficents(self, duplication_limit=2):
        """
        ピアソン類似度を算出する。

        Returns
        -------
        similarity_dict: dict


        """
        similarity_dict = dict()

        for target in self.target_user_ids:
            print(target)
            similarity_dict[target] = dict()

            target_scores = self.scores[target]
            target_mean_score = self.get_mean_score(target_scores)

            for target2 in self.user_ids:
                if target == target2:
                    continue
                target2_scores = self.scores[target2]

                # 同じ商品に対し(duplication_limit)つ以上行動を行っていないと類似度の計算を行わない。
                duplication = list(set(target_scores.keys()) & set(target2_scores.keys()))
                print(len(duplication))
                if len(duplication) > duplication_limit:
                    target2_mean_score = self.get_mean_score(target2_scores)

                    up = 0.0
                    down = 0.0
                    down2 = 0.0
                    for product in duplication:
                        up += (self.scores[target][product] - target_mean_score) * (self.scores[target2][product] - target2_mean_score)
                        down += (self.scores[target][product] - target_mean_score) ** 2
                        down2 += (self.scores[target2][product] - target2_mean_score) ** 2
                    similarity = up / (math.sqrt(down) * math.sqrt(down2))
                    similarity_dict[target][target2] = similarity

        return similarity_dict

    def predict_recommendation_scores(self, similarities):
        target_users = similarities.keys()
        recommendation_scores = dict()
        for target in target_users:
            print(target)
            recommendation_scores[target] = dict()
            for product in self.product_ids:
                up = 0.0
                down = 0.0
                for target2, similarity in similarities[target].items():
                    down += abs(similarity)
                    if product in self.scores[target2].keys():
                        target2_scores = self.scores[target2]
                        target2_mean_score = self.get_mean_score(target2_scores)
                        score = self.scores[target2][product]
                        up += (score - target2_mean_score) * similarity
                if ((down == 0) or (up == 0)):
                    continue
                recommendation_score = up / down
                recommendation_scores[target][product] = recommendation_score

        return recommendation_scores
