import csv
from cf import DataProcessor
import pandas as pd
import time
import sys
from mf import MatrixFactorization
import numpy as np

if __name__=="__main__":
    data = []
    with open("train_B.tsv","r") as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            data.append(row)
    data.pop(0)

    # ユーザー、プロダクトのクラスター対応辞書オブジェクトを作る用
    df = pd.read_csv("train_B.tsv", sep='\t')

    start = time.time()
    processor = DataProcessor(data)

    user_cluster_dict = processor.make_user_cluster_dict(df)
    product_cluster_dict = processor.make_product_cluster_dict(df)

    matrix = processor.make_matrix_for_CF(user_cluster_dict,product_cluster_dict)

    # 評価値行列の確認用なので、必要なかったらコメントアウトしてください
    print(matrix[0])


    """
        以下熊田さんのMatrix Factorization
        今回は荻野さんのMatrix Factorizationを採用するとのことで、
        こちらは使わないのでコメントアウトしておきます。"""

    matrix_size = processor.get_max_ids()
    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    mf=MatrixFactorization()

    start = time.time()
    P,Q = mf.factorize(matrix)
    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    print(np.dot(P,Q))


    with open("user_vector_A.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in P:
            writer.writerow(row)
    with open("dproduct_vector_A.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in Q:
            writer.writerow(row)
