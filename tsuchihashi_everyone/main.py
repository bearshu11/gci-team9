import csv
from cf import DataProcessor
from cf import MatrixFactorization
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
    
    df = pd.read_csv("train/train_A.tsv", sep='\t')

    start = time.time()
    processor = DataProcessor(data)
    
    user_cluster_dict = processor.make_user_cluster_dict(df)
    product_cluster_dict = processor.make_product_cluster_dict(df)
    
    matrix = processor.make_matrix_for_CF(user_cluster_dict,product_cluster_dict)
    print(matrix)
    sys.exit()
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
