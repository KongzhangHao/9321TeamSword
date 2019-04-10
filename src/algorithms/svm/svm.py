import csv
from src.algorithms.svm.helper import *
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import pandas as pd

headers = []


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

    result.append(row[1])
    nums.append(len(result))
    headers += ["a2"]


    if row[2] == 1:
        result += [1, 0, 0, 0]
    elif row[2] == 2:
        result += [0, 1, 0, 0]
    elif row[2] == 3:
        result += [0, 0, 1, 0]
    elif row[2] == 4:
        result += [0, 0, 0, 1]
    else:
        raise Exception("Thal is invalid")
    nums.append(len(result))
    headers += ["a3_1", "a3_2", "a3_3", "a3_4"]



    if row[3] <= 130:
        result += [1, 0]
    else:
        result += [0, 1]
    nums.append(len(result))
    headers += ["a4_0", "a4_1"]


    if row[4] <= 243:
        result += [1, 0]
    else:
        result += [0, 1]
    nums.append(len(result))
    headers += ["a5_0", "a5_1"]


    result.append(row[5])
    nums.append(len(result))
    headers += ["a6"]


    if row[6] == 0:
        result += [1, 0, 0]
    elif row[6] == 1:
        result += [0, 1, 0]
    elif row[6] == 2:
        result += [0, 0, 1]
    else:
        raise Exception("Thal is invalid")
    nums.append(len(result))
    headers += ["a7_0", "a7_1", "a7_2"]



    if row[7] <= 153:
        result += [1, 0]
    else:
        result += [0, 1]
    nums.append(len(result))
    headers += ["a8_0", "a8_1"]



    result.append(row[8])
    nums.append(len(result))
    headers += ["a9_0", "a9_1"]


    # if row[9] <= 0.8:
    #     result += [1, 0]
    # else:
    #     result += [0, 1]
    row[9] = int(row[9])
    if row[9] == 0:
        result += [1, 0, 0, 0, 0, 0, 0]
    elif row[9] == 1:
        result += [0, 1, 0, 0, 0, 0, 0]
    elif row[9] == 2:
        result += [0, 0, 1, 0, 0, 0, 0]
    elif row[9] == 3:
        result += [0, 0, 0, 1, 0, 0, 0]
    elif row[9] == 4:
        result += [0, 0, 0, 0, 1, 0, 0]
    elif row[9] == 5:
        result += [0, 0, 0, 0, 0, 1, 0]
    elif row[9] == 6:
        result += [0, 0, 0, 0, 0, 0, 1]
    else:
        raise Exception("Thal is invalid")
    nums.append(len(result))
    headers += ["a10_0", "a10_1", "a10_2", "a10_3", "a10_4", "a10_5", "a10_6"]



    if row[10] == 1:
        result += [1, 0, 0]
    elif row[10] == 2:
        result += [0, 1, 0]
    elif row[10] == 3:
        result += [0, 0, 1]
    else:
        raise Exception("Thal is invalid")
    nums.append(len(result))
    headers += ["a11_1", "a11_2", "a11_3"]



    if row[11] == 0:
        result += [1, 0, 0, 0]
    elif row[11] == 1:
        result += [0, 1, 0, 0]
    elif row[11] == 2:
        result += [0, 0, 1, 0]
    elif row[11] == 3:
        result += [0, 0, 0, 1]
    else:
        raise Exception("Thal is invalid")
    nums.append(len(result))
    headers += ["a12_0", "a12_1", "a12_2", "a12_3"]



    if row[-2] == 3:
        result += [1, 0, 0]
        # row = row[:-2] + [1, 0, 0] + [row[-1]]
    elif row[-2] == 6:
        result += [0, 1, 0]
        # row = row[:-2] + [0, 1, 0] + [row[-1]]
    elif row[-2] == 7:
        result += [0, 0, 1]
        # row = row[:-2] + [0, 0, 1] + [row[-1]]
    else:
        raise Exception("Thal is invalid")
    nums.append(len(result))
    headers += ["a13_3", "a13_6", "a13_7"]



    if row[-1] != 0:
        result.append(1)
    else:
        result.append(0)
    nums.append(len(result))
    # print(nums)
    return result


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


def print_rows(rows):
    for row in rows:
        print("\t\t".join(map(lambda x: str(x), row)))


def print_row_medium(rows, num):
    values = []
    for row in rows:
        values.append(row[num])
    print(sorted(values)[int(len(values) / 2)])


def build_svm(rows):
    x_train = []
    y_train = []

    for row in rows:
        x_train.append(row[:-1])
        y_train.append(row[-1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    parameters = {'gamma': 'auto', 'C': 20, 'kernel': 'linear', 'degree': 1, 'coef0': 0.0}
    clf = train_svm(parameters, x_train, y_train)
    # weights = clf.coef_[0]

    return clf


def test_accuracy(clf, path):
    rows = []
    succss = 0
    modified_success = [0 for i in range(35)]
    modified_total = [0 for i in range(35)]
    total = 0

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if is_clean_data(row):
                row = preprocess(row)
                total += 1
                if clf.predict([row[:-1]])[0] == row[-1]:
                    succss += 1
                #
                # for i in range(35):
                #     if row[i] == 0:
                #         continue
                #     modified_total[i] += 1
                #     row_copy = deepcopy(row)
                #     row_copy[i] = 0
                #     if clf.predict([row_copy[:-1]])[0] == row[-1]:
                #         modified_success[i] += 1

    # for i in range(35):
    #     modified_success[i] = modified_success[i] / modified_total[i]
    #
    # print(succss / total * 100)
    # print(modified_success)
    return succss / total * 100


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def svm_strategy(data_path):
    rows = read_csv(data_path)
    # print_row_medium(rows, 9)
    # print_rows(rows)

    clf = build_svm(rows)

    print(test_accuracy(clf, data_path), "%")
    weights = clf.coef_[0]
    weights = list(map(lambda x:abs(x), weights))
    print(weights)
    result = {}
    for i in range(35):
        result[i] = weights[i]

    for key in sorted(result.keys(), key=lambda x: result[x], reverse=True):
        print(headers[key], "\t\t", result[key])

    return


svm_strategy('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')