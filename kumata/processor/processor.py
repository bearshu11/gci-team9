import pandas as pd

def make_matrix(filename, params=[1,2,3,4]):
    '''
    与えられたデータから、協調フィルタリングに用いることができるマトリクスを生成する。

    Parameter
    ---------
    filename: データを保存しているfileの名前(.tsv, .csvのみ)
    params: ユーザーのアイテムに対する評価値を算出するために仮で設定したパラメータ
    '''

    # filenameが .tsv, .csv 以外なら処理しない。
    ext = filename.split('.')[-1]
    if ext == "csv":
        sep = ','
    elif ext == "tsv":
        sep = '\t'
    else:
        print("please change file extension")
        return

    df = pd.read_csv(filename, sep=sep)

    # 新しく作るmatrixのcolumnの名前を定義
    new_columns = ["user_id"]
    new_columns.extend(list(set(df.product_id)))
    # user_idをすべて抽出
    user_ids = set(df.user_id)

    # 新しく作るmatrixの１行１行を保存するlist
    row_data = []

    for user_id in user_ids:
        # user_idごとにクリック等をしたことのあるitem_idを抽出
        per_df = df.loc[df.user_id == user_id]
        per_product_ids = set(per_df.product_id)

        # 新しく作るmatrixの１行分を保存するSeries
        new_row = pd.Series(index=new_columns)
        new_row["user_id"] = user_id

        # それぞれのアイテムに対する評価値を算出する
        # TODO:適切な評価方法に設定する。
        for per_product_id in per_product_ids:
            ex_df = per_df[per_df["product_id"] == per_product_id]
            n0 = len(ex_df[ex_df["event_type"] == 0].index)
            n1 = len(ex_df[ex_df["event_type"] == 1].index)
            n2 = len(ex_df[ex_df["event_type"] == 2].index)
            n3 = len(ex_df[ex_df["event_type"] == 3].index)
            value = 1*n1 + 2*n2 + 3*n0 + 4*n3
            new_row[per_product_id] = value

        new_row = new_row.fillna(0)
        row_data.append(new_row)
        # XXX:処理時間がかかるため、とりあえず1行のみの処理
        break

    new_df = pd.DataFrame.from_records(row_data)
    new_filename = filename.split(".")[0].split("/")[-1] + "_new.csv"
    new_df.to_csv(new_filename)

if __name__ == '__main__':
    make_matrix("data/train/train_A.tsv")
