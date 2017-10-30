#!usr/bin/python
# -*- coding:utf-8 -*-

import csv
from processor import DataProcessor
from mf import MatrixFactorization
from recommendation import Recommendation
from tools import *
import time
import pandas as pd
import numpy as np


if __name__=="__main__":

    # XXX:以下の範囲内を編集すべし
    # ----------------------------------------------------------------
    # ---ファイルの読み込み先の設定---

    # TODO:元データのfileの場所(行数に関してはデータを小さくしてもよい)
    data_filename = "./sample_data/min_C2.csv"

    # TODO:user_idとクラスターの関係を記したfileの場所(必要なければNoneを代入)
    #user_cluster_filename = "./sample_data/prepared_data/user_cluster_C.csv"
    user_cluster_filename = None

    # TODO:product_idとクラスターの関係を記したfileの場所(必要なければNoneを代入)
    product_cluster_filename = None

    # TODO:提出すべきuser_idが記してあるfile(test.tsv)の場所と対象のカテゴリー
    test_filename = "../../data/test.tsv"
    category = "C"

    # ---ファイル出力先の設定---

    # TODO:MF後の user_id x 特徴量 のベクトルの保存先
    user_vector_filename = "./sample_data/min_user_vector_C.csv"

    # TODO:MF後の product_id x 特徴量 のベクトルの保存先
    product_vector_filename = "./sample_data/min_product_vector_C.csv"

    # ----------------------------------------------------------------
    # XXX:以下変更不要

    data = measure_time(load_raw_file, data_filename)

    users_cluster = None
    if not user_cluster_filename is None:
        users_cluster = measure_time(load_users_cluster_file, user_cluster_filename)

    products_cluster = None
    if not product_cluster_filename is None:
        products_cluster = measure_time(load_products_cluster_file, product_cluster_filename)

    target_user_ids = get_target_user_ids(test_filename, category)


    # ---評価値の表を作成する処理---
    print("computing value matrix...")
    start = time.time()

    # 評価値を算出するためのクラスのインスタンス
    value_computer = ValueComputer()

    processor = DataProcessor(data, value_computer, users_cluster, products_cluster)
    user_ids = processor.user_ids
    product_ids = processor.product_ids

    matrix = processor.make_matrix_for_CF()

    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")
    print()

    # ---Matrix Factorization---
    print("matrix factorization...")
    start = time.time()

    mf = MatrixFactorization()

    print(str(len(user_ids)), str(len(product_ids)))
    nP,nQ = mf.factorize(matrix, user_ids, product_ids, K=50, threshold=20000, steps=500)
    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    print(nP[0], nQ[0])
    np.savetxt(user_vector_filename, nP, delimiter=",")
    np.savetxt(product_vector_filename, nQ, delimiter=",")


    # MFを行わずに類似度を算出する処理(今回は不要)
    # print("finding recommendation...")
    # start = time.time()
    #
    # rc = Recommendation(matrix, target_user_ids, user_ids, product_ids)
    #
    # similarities = rc.get_correlation_coefficents()
    # print(similarities[231])
    # recommendation_scores = rc.predict_recommendation_scores(similarities)
    # print(recommendation_scores[231])
    #
    # dif = time.time() - start
    # print("time:{}".format(dif)+"[sec]")
    # print()
