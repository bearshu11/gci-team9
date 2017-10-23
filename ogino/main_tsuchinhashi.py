#!usr/bin/python
# -*- coding:utf-8 -*-

import csv
from cf import DataProcessor
import pandas as pd
import time
import sys

if __name__=="__main__":
    print(time.ctime()),
    print("start main")
    data = []
    with open("../data/train/train_A.tsv","r") as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            data.append(row)
    data.pop(0)
    
    # ユーザー、プロダクトのクラスター対応辞書オブジェクトを作る用
    df = pd.read_csv("../data/train/train_A.tsv", sep='\t')

    start = time.time()
    processor = DataProcessor(data)
    
    user_cluster_dict = processor.make_user_cluster_dict(df)
    product_cluster_dict = processor.make_product_cluster_dict(df)
    
    print(time.ctime()),
    print("start make_matrix_for_CF")
    matrix = processor.make_matrix_for_CF(user_cluster_dict,product_cluster_dict)
    print(time.ctime()),
    print("end make_matrix_for_CF")
    
    # 評価値行列の確認用なので、必要なかったらコメントアウトしてください
    print(matrix[0])
    sys.exit()

