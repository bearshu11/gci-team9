import numpy as np
from datetime import datetime as dt

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

    def compute_value(self, user_id, product_id, actions, user2cluster, product2cluster):
        value = 3
        cluster = user2cluster[user_id]

        if (cluster == 0) or (cluster == 2) or (cluster == 9):
            ca_values = {0:0.00001, 2:0.0001, 9:0.001}
            pd_values = {0:0.01, 2:0.01, 9:0.01}
            for action in actions:
                if action[0] == 0:
                    value += ca_values[cluster]
                elif action[0] == 1:
                    value += pd_values[cluster]
        elif (cluster == 4) or (cluster == 5) or (cluster == 8):
            ca_values = {4:0.01, 5:0.05, 8:0.05}
            pd_values = {4:0.1, 5:0.1, 8:0.1}
            for action in actions:
                if action[0] == 0:
                    value += ca_values[cluster]
                elif action[0] == 1:
                    value += pd_values[cluster]

        elif (cluster == 1) or (cluster == 7):
            ca_values = {1:0.001, 7:0.003}
            pd_values = {1:0.01, 7:0.01}
            cv_values = {1:0.3, 7:1.0} # 10はcvが存在しない
            for action in actions:
                if action[0] == 0:
                    value += ca_values[cluster]
                elif action[0] == 1:
                    value += pd_values[cluster]
                elif action[0] == 3:
                    value += cv_values[cluster]

        elif (cluster == 3) or (cluster == 10):
            ca_values = {3:-0.001, 10:-0.003}
            pd_values = {3:0.1, 10:0.1}
            cv_values = {3:1} # 10はcvが存在しない
            for action in actions:
                if action[0] == 0:
                    value += ca_values[cluster]
                elif action[0] == 1:
                    value += pd_values[cluster]
                elif action[0] == 3:
                    value += cv_values[cluster]

        elif cluster == 6:
            ca_values = {6:0.01}
            pd_values = {6:0.001}
            cv_values = {6:1}
            for action in actions:
                if action[0] == 0:
                    value += ca_values[cluster]
                elif action[0] == 1:
                    value += pd_values[cluster]
                elif action[0] == 3:
                    value += cv_values[cluster]

        return value
