import csv
from processor import DataProcessor
from mf import MatrixFactorization
import time

if __name__=="__main__":
    data = []
    with open("../data/train/min_D.csv","r") as f:
        reader = csv.reader(f)

        for row in reader:
            if row[0] == "user_id":
                continue
            data.append(row)

    start = time.time()
    processor = DataProcessor(data)
    matrix = processor.make_matrix_for_CF()
    user_ids, product_ids = processor.get_ids()
    user_ids = sorted(user_ids)
    product_ids = sorted(product_ids)

    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    mf = MatrixFactorization()

    start = time.time()
    nP,nQ = mf.factorize(matrix, user_ids, product_ids, 50)
    dif = time.time() - start
    print("time:{}".format(dif)+"[sec]")

    print(nP[0], nQ[0])

    with open("../data/train/min_user_vector_D.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for user_id, values in zip(user_ids, nP):
            row = [user_id].extend(values)
            writer.writerow(row)
    with open("../data/train/min_product_vector_D.csv","w") as f:
        writer = csv.writer(f, delimiter=',')
        for product_id, values in zip(product_ids, nQ):
            row = [product_id].extend(values)
            writer.writerow(row)
