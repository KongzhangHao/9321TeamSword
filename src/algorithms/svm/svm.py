import csv

def is_clean_data(row):
    if "?" in row:
        return False
    return True


# [0, 2, 3, 7, 9, 11, 12, 15, 17, 18, 25, 28, 32, 35, 36]
def preprocess(row):
    global headers
    nums = [0]
    result = []
    # Cast each term to int
    row = list(map(lambda x: float(x), row))
    # Reassign final result
    if row[0] < 40:
        result += [1, 0]
    else:
        result += [0, 1]
    nums.append(len(result))
    headers += ["a1_0", "a1_1"]

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