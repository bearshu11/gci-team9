import numpy as np
from tools import *
import random
import pandas as pd

def _id_int2str(int_id, category):
    return "{0:{fill}7}_".format(int_id, fill="0", align="right") + str(category)


if __name__ == "__main__":

    print("loading....")

    user_vectors = np.loadtxt("../../data/train/min_user_vector_D.csv", delimiter=",")
    # user_vectors = []
    # with open("../data/train/min_user_vector_D.csv","r") as f:
    #     reader = csv.reader(f)
    #
    #     for row in reader:
    #         user_vectors.append(row)

    product_vectors = np.loadtxt("../../data/train/min_product_vector_D.csv", delimiter=",")
    # product_vectors = []
    # with open("../data/train/min_product_vector_D.csv","r") as f:
    #     reader = csv.reader(f)
    #
    #     for row in reader:
    #         product_vectors.append(row)

    filename = "../../data/test.tsv"
    category = "D"
    target_user_ids = get_target_user_ids(filename, category)

    products_cluster_df = pd.read_csv("../../data/train/product_cluster_D.csv")
    products_cluster2 = products_cluster_df[products_cluster_df["cluster"] == 2].values.tolist()


    products_index = []
    with open("../../data/train/min_product_index_D.csv","r") as f:
        reader = csv.reader(f)

        for row in reader:
            products_index.append(int(row[0]))

    users_index = []
    with open("../../data/train/min_user_index_D.csv","r") as f:
        reader = csv.reader(f)

        for row in reader:
            users_index.append(int(row[0]))

    print("recommending...")

    random.seed(0)

    recommendation = dict()
    for target_user_id in target_user_ids:
        row = [0] * len(product_vectors)
        str_recommend_user_id = _id_int2str(target_user_id,"D")
        recommendation[str_recommend_user_id] = []
        if target_user_id in users_index:
            target_user_index = users_index.index(target_user_id)
            for product_index, product_vector in enumerate(product_vectors):
                row[product_index] = np.dot(user_vectors[target_user_index], product_vector)
            recommend_product_indices = np.argsort(np.array(row))[::-1]
            for recommend_product_index in recommend_product_indices:
                str_recommend_product_id = _id_int2str(products_index[recommend_product_index],"d")

                recommendation[str_recommend_user_id].append(str_recommend_product_id)
                if len(recommendation[str_recommend_user_id]) > 21:
                    break
        else:
            recommend_list = random.sample(products_cluster2, 22)
            recommendation[str_recommend_user_id] = recommend_list
            print("not exist {}".format(str_recommend_user_id))

    with open("../../data/submit_D.tsv","w") as f:
        writer = csv.writer(f, delimiter='\t')
        for user_id, product_ids in recommendation.items():
            for i, product_id in enumerate(product_ids):
                writer.writerow([user_id, product_id, i])
