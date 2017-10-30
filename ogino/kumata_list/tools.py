#!usr/bin/python
# -*- coding:utf-8 -*-

import csv
import time
import pandas as pd

# XXX:以下の範囲内のメソッド部分を編集すべし
# ----------------------------------------------------------------
class ValueComputer():
    """
    評価値を計算するためのクラス
    """
    def compute_value(self, user_id, product_id, actions, user_cluster_no, product_cluster_no):
        """
        評価値を算出するメソッド

        Params
        -------
        user_id: int
        例："000034_A" => 34

        product_id: int
        例："000103_a" => 103

        actions: list
        user_idのproduct_idに対する行動のリスト
        actions[i][0]: int "event_type"
        actions[i][1]: int "ad"
        actions[i][2]: datetime "time_stamp"
        (iは 0<=i<len(actions) の整数)

        user_cluster_no: int
        ユーザーのクラスター番号

        product_cluster_no: int
        プロダクトのクラスター番号

        """
        for action in actions:
            if action[0] == 3 and action[1] == 1:
                return 5.0
        for action in actions:
            if action[0] == 3 and action[1] == 0:
                return 4.5
        score = 3.0
        for action in actions:
            if action[0] == 1:
                score += 0.3
            elif action[0] == 2:
                score += 0.3
        return min(score, 4.5)

# ----------------------------------------------------------------
# XXX:以下変更不要


def load_raw_file(filename):
    """
    train_*.tsv(与えられたデータそのまま)を読み込むメソッド
    """

    print("loading raw data...")

    data = []
    with open(filename,"r") as f:
        if filename.split(".")[-1] == "tsv":
            reader = csv.reader(f, delimiter='\t')
        elif filename.split(".")[-1] == "csv":
            reader = csv.reader(f)

        for row in reader:
            data.append(row)
    data.pop(0)

    return data

def load_users_cluster_file(filename):
    """
    ユーザーとクラスター番号の対応を記したファイルを読み込むメソッド
    """

    print("loading users' cluster number...")

    users_cluster = []
    with open(filename,"r") as f:
        reader = csv.reader(f)

        for row in reader:
            if row[0] == "user_id":
                continue
            users_cluster.append(row)

    return users_cluster

def load_products_cluster_file(filename):
    """
    プロダクトとクラスター番号の対応を記したファイルを読み込むメソッド
    """

    print("loading products' cluster number...")

    products_cluster = []
    with open(filename,"r") as f:
        reader = csv.reader(f)

        for row in reader:
            if row[0] == "product_id":
                continue
            products_cluster.append(row)

    return products_cluster

def measure_time(func, filename):
    """
    ファイルを読み込むメソッドに関して、時間を図るメソッド

    Params
    --------
    func: 関数
    load_raw_file, load_users_cluster_file, load_products_cluster_file のいずれかが対象
    filename: str
    読み込むファイルの場所
    """
    start = time.time()
    result = func(filename)
    dif_time = time.time() - start
    print("time:{}".format(dif_time)+"[sec]")
    print()
    return result

def id_str2int(str_id):
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

def id_int2str(int_id, category):
    """
    int型に変換されたuser_idまたはproduct_idを元の形に戻す

    Params
    -------
    int_id: int

    category: str
    user_idのとき: 大文字 "A","B","C","D"
    product_idのとき: 小文字 "a","b","c","d"
    """

    if category.islower():
        return "{0:{fill}8}_".format(int_id, fill="0", align="right") + str(category)
    else:
        return "{0:{fill}7}_".format(int_id, fill="0", align="right") + str(category)

def get_target_user_ids(filename, category):
    """
    test.tsvから提出すべきuser_idを抽出する。
    """
    test_df = pd.read_csv(filename, delimiter="\t")
    before_target_user_ids = test_df[test_df['user_id'].map(lambda x: x[-1]) == category].values.tolist()
    target_user_ids = []
    for item in before_target_user_ids:
        target_user_ids.append(id_str2int(item[0]))
    return target_user_ids
