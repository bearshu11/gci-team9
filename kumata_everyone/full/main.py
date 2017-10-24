import csv
from processor import DataProcessor
from mf import MatrixFactorization
from recommendation import Recommendation
from tools import *
import time
import pandas as pd


if __name__=="__main__":
    # ----------------------------------------------------------------
    # ファイル読み込み
    filename = "../../data/train/train_D.tsv"
    data = measure_time(load_raw_file, filename)

    filename = "../../data/train/user_cluster_D.csv"
    users_cluster = measure_time(load_users_cluster_file, filename)

    filename = ""
    products_cluster = measure_time(load_products_cluster_file, filename)

    filename = "../../data/test.tsv"
    category = "D"
    target_user_ids = get_target_user_ids(filename, category)

    # ----------------------------------------------------------------


    value_computer = ValueComputer()

    print("computing value matrix...")

    start = time.time()

    processor = DataProcessor(data, value_computer, users_cluster, products_cluster)
    matrix = processor.make_matrix_for_CF()
    matrix_size = processor.get_max_ids()

    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")
    print()

    # print("finding recommendation...")
    #
    # start = time.time()
    #
    #
    # user_ids, product_ids = processor.get_ids(selector="all")
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

    print("matrix factorization...")

    mf = MatrixFactorization()

    start = time.time()
    nP,nQ = mf.factorize(matrix, matrix_size, K=50)
    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    with open("./time.txt", "w") as f:
        f.write("time:{}".format(dif)+"[sec]")

    print(nP[0], nQ[0])

    with open("../../data/train/user_vector_D.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in nP:
            writer.writerow(row)
    with open("../../data/train/product_vector_D.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in nQ:
            writer.writerow(row)
