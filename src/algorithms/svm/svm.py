import csv

def read_csv(path):
    rows = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if is_clean_data(row):
                rows.append(preprocess(row))
            else:
                continue
    return rows

def svm_strategy(data_path):
    rows = read_csv(data_path)

svm_strategy('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')