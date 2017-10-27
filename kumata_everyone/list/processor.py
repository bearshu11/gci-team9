import numpy as np
from datetime import datetime as dt

class DataProcessor():
    """
    MFをするための評価値の表(dict型)を作成するクラス
    """
    
    def __init__(self, data, value_computer, users_cluster, products_cluster):
        """
        イニシャライザ

        Params
        ------
        data: list
        データの読み込み方は、main.py参照

        value_computer: object
        評価値を計算するメソッド(compute_value)を持ったクラスのオブジェクト

        users_cluster: list
        user_idとcluster番号の対応関係を記したlist

        products_cluster: list
        product_idとcluster番号の対応関係を記したlist

        """
        self.data = data
        self.value_computer = value_computer
        self.user_ids, self.product_ids = self.get_ids()

        if not users_cluster is None:
            self.users_cluster = [-1] * len(users_cluster)
            for item in users_cluster:
                int_user_id = self._id_str2int(item[0])
                self.users_cluster[int_user_id] = int(item[1])
        else:
            self.users_cluster = None

        if not products_cluster is None:
            self.products_cluster = [-1] * len(products_cluster)
            for item in products_cluster:
                int_product_id = self._id_str2int(item[0])
                self.products_cluster[int_product_id] = int(item[1])
        else:
            self.products_cluster = None

    def _id_str2int(self, str_id):
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



    def get_ids(self, selector="all"):
        user_ids = []
        product_ids = []
        for row in self.data:
            user_id = int(row[0])
            product_id = int(row[1])
            user_ids.append(user_id)
            product_ids.append(product_id)

        if selector == "all":
            return list(set(user_ids)), list(set(product_ids))
        elif selector == "user":
            return list(set(user_ids))
        elif selector == "product":
            return list(set(product_ids))


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
            user_id = int(row[0])
            product_id = int(row[1])
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
            user_id = int(row[0])
            product_id = int(row[1])

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

            user_cluster_no = None
            if not self.users_cluster is None:
                user_cluster_no = self.users_cluster[user_id]

            for product_id, actions in user_actions.items():
                product_cluster_no = None
                if not self.products_cluster is None:
                    product_cluster_no = self.products_cluster[product_id]

                # 評価値計算
                value = self.value_computer.compute_value(user_id, product_id, actions, user_cluster_no,  product_cluster_no)

                cf_dict[user_id][product_id] = value

        return cf_dict
