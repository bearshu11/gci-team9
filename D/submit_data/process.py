import pandas as pd

ogino_df = pd.read_csv("./submit_D_ogino.tsv", delimiter="\t", header=None)
full_df = pd.read_csv("./submit_data.tsv", delimiter="\t", header=None)

D_ogino_df = ogino_df[ogino_df[0].map(lambda x: x[-1]) == "D"]
non_D_full_df = full_df[full_df[0].map(lambda x: x[-1]) != "D"]

submit_df = pd.concat([D_ogino_df,non_D_full_df])

submit_df.to_csv("./submit_ogino_data.tsv", sep="\t", index=False, header=None)
