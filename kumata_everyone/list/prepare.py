import pandas as pd
import csv
from tools import id_str2int

if __name__ == "__main__":
    # TODO:元データからデータを小さくする(行数を減らす)処理（必要であれば）
    # ----------------------------------------------------------------

    df = pd.read_csv("../../data/train/train_D.tsv", sep="\t")
    # user_df = pd.read_csv("../../data/train/clustered_user_with_cv_D.csv")
    product_df = pd.read_csv("../../data/train/clustered_product_without_cv_D.csv")
    clustered_df = pd.merge(df, product_df[["product_id","cluster"]], on="product_id")
    min_df = clustered_df[clustered_df["cluster"] != 0].reset_index(drop=True)

    # ----------------------------------------------------------------
    # XXX:ここまでの出力形式
    # min_df: pandas.DataFrame
    # header: ["user_id", "product_id", "event_type", "ad" ,"time_stamp", ("cluster")]


    # XXX:以下の範囲を編集すべし
    # ----------------------------------------------------------------

    # TODO:元データから小さくしたデータの出力先
    min_df_filename = "./sample_data/min_D2.csv"

    # TODO:user_idと評価値の表におけるindexの対応の出力先(.csv)
    user_index_filename = "./sample_data/min_user_index_D.csv"

    # TODO:product_idと評価値の表におけるindexの対応の出力先(.csv)
    product_index_filename = "./sample_data/min_product_index_D.csv"

    # ----------------------------------------------------------------
    # XXX:以下変更不要


    min_df.user_id = min_df.user_id.apply(id_str2int)
    min_df.product_id = min_df.product_id.apply(id_str2int)

    user_list = min_df.user_id.values.tolist()
    user_ids = list(set(user_list))
    user_dict = dict()
    for i, user in enumerate(user_ids):
        user_dict[user] = i

    product_list = min_df.product_id.values.tolist()
    product_ids = list(set(product_list))
    product_dict = dict()
    for i, product in enumerate(product_ids):
        product_dict[product] = i

    def user_ind(id):
        return user_dict[id]

    def product_ind(id):
        return product_dict[id]

    min_df.user_id = min_df.user_id.apply(user_ind)
    min_df.product_id = min_df.product_id.apply(product_ind)

    min_df.to_csv(min_df_filename, index=False)

    with open(user_index_filename,"w") as f:
        writer = csv.writer(f, delimiter=',')
        for user_id in user_ids:
            writer.writerow([user_id])

    with open(product_index_filename,"w") as f:
        writer = csv.writer(f, delimiter=',')
        for product_id in product_ids:
            writer.writerow([product_id])
