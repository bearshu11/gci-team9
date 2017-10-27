import pandas as pd
import csv

def _id_str2int(str_id):
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

if __name__ == "__main__":
    df = pd.read_csv("../../data/train/train_D.tsv", sep="\t")
    user_df = pd.read_csv("../../data/train/clustered_user_with_cv_D.csv")
    product_df = pd.read_csv("../../data/train/clustered_product_without_cv_D.csv")

    clustered_df = pd.merge(df, product_df[["product_id","cluster"]], on="product_id")

    min_df = clustered_df[clustered_df["cluster"] != 0].reset_index(drop=True)

    min_df.user_id = min_df.user_id.apply(_id_str2int)
    min_df.product_id = min_df.product_id.apply(_id_str2int)

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

    min_df.to_csv("../../data/train/min_D2.csv", index=False)

    with open("../../data/train/min_user_index_D.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for user_id in user_ids:
            writer.writerow([user_id])

    with open("../../data/train/min_product_index_D.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for product_id in product_ids:
            writer.writerow([product_id])
