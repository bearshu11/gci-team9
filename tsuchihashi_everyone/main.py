import csv
from cf import DataProcessor
import pandas as pd
import time
import sys

if __name__=="__main__":
    data = []
    with open("train/train_A.tsv","r") as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            data.append(row)
    data.pop(0)
    
    # ユーザー、プロダクトのクラスター対応辞書オブジェクトを作る用
    df = pd.read_csv("train/train_A.tsv", sep='\t')

    start = time.time()
    processor = DataProcessor(data)
    
    user_cluster_dict = processor.make_user_cluster_dict(df)
    product_cluster_dict = processor.make_product_cluster_dict(df)
    
    matrix = processor.make_matrix_for_CF(user_cluster_dict,product_cluster_dict)
    
    # 評価値行列の確認用なので、必要なかったらコメントアウトしてください
    print(matrix[0])
    sys.exit()
    
    """
        以下熊田さんのMatrix Factorization
        今回は荻野さんのMatrix Factorizationを採用するとのことで、
        こちらは使わないのでコメントアウトしておきます。
    
    matrix_size = processor.get_max_ids()
    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    mf = MatrixFactorization()

    start = time.time()
    nP,nQ = mf.factorize(matrix, matrix_size, 3, steps=1)
    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    print(nP[0], nQ[0])

    with open("data/train/user_vector_A.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in nP:
            writer.writerow(row)
    with open("data/train/product_vector_A.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in nQ:
            writer.writerow(row)
            
    """