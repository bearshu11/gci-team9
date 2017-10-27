import csv
from cf import DataProcessor
import collaborative_filtering as cf

if __name__=="__main__":
    data = []
    with open("../data/train/train_D.tsv","r") as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            data.append(row)
    data.pop(0)

    processor = DataProcessor(data)

    user_cluster_dict = dict()
    with open("../data/train/user_cluster_D.csv","r") as f:
        reader = csv.reader(f)

        for row in reader:
            if row[0] == "user_id":
                continue
            user_cluster_dict[processor.id_str2int(row[0])] = row[1]

    # ユーザー、プロダクトのクラスター対応辞書オブジェクトを作る用
    # df = pd.read_csv("train/train_A.tsv", sep='\t')


    # user_cluster_dict = processor.make_user_cluster_dict(df)
    # product_cluster_dict = processor.make_product_cluster_dict(df)

    # matrix = processor.make_matrix_for_CF(user_cluster_dict,product_cluster_dict)
    matrix = processor.make_matrix_for_CF(user_cluster_dict)
    print(matrix[0])
