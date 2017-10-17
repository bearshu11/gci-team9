import csv
from cf import DataProcessor
from cf import MatrixFactorization
import time

if __name__=="__main__":
    data = []
    with open("train/train_A.tsv","r") as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            data.append(row)
    data.pop(0)

    start = time.time()
    processor = DataProcessor(data)
    matrix = processor.make_matrix_for_CF()
    
    print(matrix)