import numpy as np
from tools import *
import random
import pandas as pd

if __name__ == "__main__":

    # XXX:以下の範囲を編集すべし
    # ----------------------------------------------------------------

    # ---ファイルの読み込み先の設定---

    # TODO:MF後の user_id x 特徴量 のベクトルの保存先(.csv)
    user_vector_filename = "./sample_data/min_user_vector_D.csv"

    # TODO:MF後の product_id x 特徴量 のベクトルの保存先(.csv)
    product_vector_filename = "./sample_data/min_product_vector_D.csv"

    # TODO:提出すべきuser_idが記してあるfile(test.tsv)の場所と対象のカテゴリー
    test_filename = "./sample_data/raw_data/test.tsv"
    category = "D"

    # TODO:user_idと評価値の表におけるindexの対応を記したfile(.csv)の場所
    user_index_filename = "./sample_data/min_user_index_D.csv"

    # TODO:product_idと評価値の表におけるindexの対応を記したfile(.csv)の場所
    product_index_filename = "./sample_data/min_product_index_D.csv"


    # ---ファイル出力先の設定---

    # TODO:提出するファイルの保存先(.tsv)
    submit_filename = "../submit_data/submit_D_kumata.tsv"


    # ---存在しないuser_idへの対応---

    products_cluster_df = pd.read_csv("./sample_data/prepared_data/product_cluster_D.csv")
    products_cluster2 = products_cluster_df[products_cluster_df["cluster"] == 2].values.tolist()
    random.seed(0)

    def for_non_exist_user_id(user_id):
        """
        データを減らしていた場合、提出すべきuser_idが評価値の表の中に存在しないことがある。
        そのようなことに対応するための関数

        Params
        -------
        user_id: str

        Returns
        -------
        recommend_list: list(str)
        user_idに対して、推薦する22個のproduct_idのlist
        """

        recommend_list = random.sample(products_cluster2, 22)
        return recommend_list

    # ----------------------------------------------------------------
    # XXX:以下変更不要


    print("loading....")
    user_vectors = np.loadtxt(user_vector_filename, delimiter=",")
    product_vectors = np.loadtxt(product_vector_filename, delimiter=",")

    target_user_ids = get_target_user_ids(test_filename, category)

    users_index = []
    with open(user_index_filename,"r") as f:
        reader = csv.reader(f)

        for row in reader:
            users_index.append(int(row[0]))

    products_index = []
    with open(product_index_filename,"r") as f:
        reader = csv.reader(f)

        for row in reader:
            products_index.append(int(row[0]))


    print("recommending...")
    recommendation = dict()
    for target_user_id in target_user_ids:
        row = [0] * len(product_vectors)
        str_recommend_user_id = id_int2str(target_user_id,"D")
        recommendation[str_recommend_user_id] = []
        if target_user_id in users_index:
            target_user_index = users_index.index(target_user_id)
            for product_index, product_vector in enumerate(product_vectors):
                row[product_index] = np.dot(user_vectors[target_user_index], product_vector)
            recommend_product_indices = np.argsort(np.array(row))[::-1]
            for recommend_product_index in recommend_product_indices:
                str_recommend_product_id = id_int2str(products_index[recommend_product_index],"d")

                recommendation[str_recommend_user_id].append(str_recommend_product_id)
                if len(recommendation[str_recommend_user_id]) > 21:
                    break
        else:
            recommend_list = for_non_exist_user_id(str_recommend_user_id)
            recommendation[str_recommend_user_id] = recommend_list
            print("not exist {}".format(str_recommend_user_id))

    with open(submit_filename, "w") as f:
        writer = csv.writer(f, delimiter='\t')
        for user_id, product_ids in recommendation.items():
            for i, product_id in enumerate(product_ids):
                writer.writerow([user_id, product_id, i])
