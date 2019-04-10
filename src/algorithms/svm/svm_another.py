import csv
from src.algorithms.svm.helper import *
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


# load file into dataframe and
# clean invalid data
def load_file(file_name):
    # load data into dataframe
    titles = ['age', 'sex', 'pain_type', 'blood_pressure', 'serum_cholestoral',
              'blood_sugar', 'electrocardiographic', 'maximum_heart_rate',
              'exercise', 'oldpeak', 'slope_of_peak_exercise', 'number_of_vessels',
              'thal', 'target']
    titles_backup = ['a' + str(i) for i in range(1, 15)]

    df = pd.read_csv(file_name, sep=',', header=None, names=titles_backup)

    # clean invalid rows
    df = df.dropna()
    df = df.astype(str)
    for title in titles_backup:
        df.drop(df.index[df[title] == '?'], inplace=True)
    df = df.astype(float)

    df = shuffle(df)
    return df


# pre-process data
def pre_process_data(df):
    # normalize continuous value of attributes
    for column in df:
        if column == 'a1' or column == 'a4' or column == 'a5' or column == 'a8' or column == 'a10' or column == 'a11':
            diff = max(df[column]) - min(df[column])
            df[column] = (df[column] - min(df[column])) / diff

    # split discrete value of attributes
    # attribute 3
    df['a3_1'] = df['a3'].map(lambda x: x == 1)
    df['a3_2'] = df['a3'].map(lambda x: x == 2)
    df['a3_3'] = df['a3'].map(lambda x: x == 3)
    df['a3_4'] = df['a3'].map(lambda x: x == 4)
    df = df.drop(['a3'], axis=1)

    # attribute 7
    df['a7_0'] = df['a7'].map(lambda x: x == 0)
    df['a7_1'] = df['a7'].map(lambda x: x == 1)
    df['a7_2'] = df['a7'].map(lambda x: x == 2)
    df = df.drop(['a7'], axis=1)

    # attribute 12
    df['a12_0'] = df['a12'].map(lambda x: x == 0)
    df['a12_1'] = df['a12'].map(lambda x: x == 1)
    df['a12_2'] = df['a12'].map(lambda x: x == 2)
    df['a12_3'] = df['a12'].map(lambda x: x == 3)
    df = df.drop(['a12'], axis=1)

    # attribute 13
    df['a13_3'] = df['a13'].map(lambda x: x == 3)
    df['a13_6'] = df['a13'].map(lambda x: x == 6)
    df['a13_7'] = df['a13'].map(lambda x: x == 7)
    df = df.drop(['a13'], axis=1)

    # attribute 14
    df['a14'] = df['a14'].map(lambda x: x != 0)

    df = df.astype(float)

    # re-arrange attributes columns
    df = df[['a1', 'a2', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a4', 'a5', 'a6',
             'a7_0', 'a7_1', 'a7_2', 'a8', 'a9', 'a10', 'a11', 'a12_0',
             'a12_1', 'a12_2', 'a12_3', 'a13_3', 'a13_6', 'a13_7', 'a14']]

    return df

def pre_process_data_optimised(df):
    # normalize continuous value of attributes
    for column in df:
        if column == 'a1' or column == 'a4' or column == 'a5' or column == 'a8' or column == 'a10':
            diff = max(df[column]) - min(df[column])
            df[column] = (df[column] - min(df[column])) / diff

    # split discrete value of attributes
    # attribute 3
    df['a3_1'] = df['a3'].map(lambda x: x == 1)
    df['a3_2'] = df['a3'].map(lambda x: x == 2)
    df['a3_3'] = df['a3'].map(lambda x: x == 3)
    df['a3_4'] = df['a3'].map(lambda x: x == 4)
    df = df.drop(['a3'], axis=1)

    # attribute 7
    df['a7_0'] = df['a7'].map(lambda x: x == 0)
    df['a7_1'] = df['a7'].map(lambda x: x == 1)
    df['a7_2'] = df['a7'].map(lambda x: x == 2)
    df = df.drop(['a7'], axis=1)

    # attribute 11
    df['a11_1'] = df['a11'].map(lambda x: x == 1)
    df['a11_2'] = df['a11'].map(lambda x: x == 2)
    df['a11_3'] = df['a11'].map(lambda x: x == 3)
    df = df.drop(['a11'], axis=1)

    # attribute 12
    df['a12_0'] = df['a12'].map(lambda x: x == 0)
    df['a12_1'] = df['a12'].map(lambda x: x == 1)
    df['a12_2'] = df['a12'].map(lambda x: x == 2)
    df['a12_3'] = df['a12'].map(lambda x: x == 3)
    df = df.drop(['a12'], axis=1)

    # attribute 13
    df['a13_3'] = df['a13'].map(lambda x: x == 3)
    df['a13_6'] = df['a13'].map(lambda x: x == 6)
    df['a13_7'] = df['a13'].map(lambda x: x == 7)
    df = df.drop(['a13'], axis=1)

    # attribute 14
    df['a14'] = df['a14'].map(lambda x: x != 0)

    df = df.astype(float)

    # re-arrange attributes columns
    df = df[['a1', 'a2', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a4', 'a5','a6',
             'a7_0', 'a7_1', 'a7_2', 'a8', 'a9', 'a10', 'a11_1', "a11_2", "a11_3", 'a12_0',
             'a12_1', 'a12_2', 'a12_3', 'a13_3', 'a13_6', 'a13_7', 'a14']]

    return df


def is_clean_data(row):
    if "?" in row:
        return False
    return True


def preprocess(row):
    result = []
    # Cast each term to int
    row = list(map(lambda x: int(float(x)), row))
    # Reassign final result
    if row[0] < 40:
        result += [0]
    else:
        result += [1]

    result.append(row[1])

    if row[2] == 1:
        result += [0]
    elif row[2] == 2:
        result += [1 / 3]
    elif row[2] == 3:
        result += [2 / 3]
    elif row[2] == 4:
        result += [1]
    else:
        raise Exception("Thal is invalid")

    if row[3] <= 130:
        result += [0]
    else:
        result += [1]

    if row[4] <= 243:
        result += [0]
    else:
        result += [1]

    result.append(row[5])

    if row[6] == 0:
        result += [0]
    elif row[6] == 1:
        result += [0.5]
    elif row[6] == 2:
        result += [1]
    else:
        raise Exception("Thal is invalid")

    if row[7] <= 153:
        result += [0]
    else:
        result += [1]

    result.append(row[8])

    if row[9] == 0:
        result += [0]
    elif row[9] == 1:
        result += [1 / 6]
    elif row[9] == 2:
        result += [2 / 6]
    elif row[9] == 3:
        result += [3 / 6]
    elif row[9] == 4:
        result += [4 / 6]
    elif row[9] == 5:
        result += [5 / 6]
    elif row[9] == 6:
        result += [1]
    else:
        raise Exception("Thal is invalid")

    if row[10] == 1:
        result += [0]
    elif row[10] == 2:
        result += [0.5]
    elif row[10] == 3:
        result += [1]
    else:
        raise Exception("Thal is invalid")

    if row[11] == 0:
        result += [0]
    elif row[11] == 1:
        result += [1 / 3]
    elif row[11] == 2:
        result += [2 / 3]
    elif row[11] == 3:
        result += [1]
    else:
        raise Exception("Thal is invalid")

    if row[-2] == 3:
        result += [0]
        # row = row[:-2] + [1, 0, 0] + [row[-1]]
    elif row[-2] == 6:
        result += [0.5]
        # row = row[:-2] + [0, 1, 0] + [row[-1]]
    elif row[-2] == 7:
        result += [1]
        # row = row[:-2] + [0, 0, 1] + [row[-1]]
    else:
        raise Exception("Thal is invalid")

    if row[-1] != 0:
        result.append(1)
    else:
        result.append(0)

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
                print(row)
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

    x_train = np.array(x_train * 2)
    y_train = np.array(y_train * 2)

    parameters = {'gamma': 'auto', 'C': 20, 'kernel': 'linear', 'degree': 1, 'coef0': 0.0}
    # print(x_train)
    clf = train_svm(parameters, x_train, y_train)
    weights = clf.coef_[0]
    # print(weights)

    return clf


def test_accuracy(clf, path):
    rows = []
    succss = 0
    modified_success = [0 for i in range(24)]
    modified_total = [0 for i in range(24)]
    total = 0

    df = load_file('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
    # print(pre_process_data(df))
    rows = pre_process_data_optimised(df).values.tolist()
    # rows = df.values.tolist()

    for row in rows:
        total += 1
        if clf.predict([row[:-1]])[0] == row[-1]:
            succss += 1

    return succss / total * 100


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def pre_process_data_normalisation(df):
    # normalize continuous value of attributes
    for column in df:
        diff = max(df[column]) - min(df[column])
        df[column] = (df[column] - min(df[column])) / diff

    df = df.astype(float)

    return df


def svm_strategy(data_path):
    # rows = read_csv(data_path)
    # print_row_medium(rows, 7)
    df = load_file('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
    # print(pre_process_data(df))
    rows = pre_process_data_optimised(df).values.tolist()
    # rows = df.values.tolist()

    # print(rows)
    clf = build_svm(rows)

    print(test_accuracy(clf, data_path), "%")

    indices = ['a1', 'a2', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a4', 'a5','a6',
             'a7_0', 'a7_1', 'a7_2', 'a8', 'a9', 'a10', 'a11_1', "a11_2", "a11_3", 'a12_0',
             'a12_1', 'a12_2', 'a12_3', 'a13_3', 'a13_6', 'a13_7', 'a14']
    weights = clf.coef_[0]
    weights = list(map(lambda x:abs(x), weights))
    print(weights)
    result = {}
    for i in range(len(weights)):
        result[i] = weights[i]

    for key in sorted(result.keys(), key=lambda x: result[x], reverse=True):
        print("%s,%s" % (indices[key], result[key]))

    return


svm_strategy('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
