import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def predict(test_data, training_data, k):
    training_data['distance'] = 0
    print("hello")
    temp = (training_data.iloc[:, :-2] - test_data) ** 2
    training_data['distance'] = temp.sum(1) ** 1 / 2
    knn = training_data.sort_values(by='distance')[:k]
    print("aaa")
    training_data.drop('distance', axis=1, inplace=True)
    return knn.iloc[:, -2].value_counts().index[0]

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
    df = df[['a1', 'a2', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a4', 'a5','a6',
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


def pre_process_data_normalisation(df):
    # normalize continuous value of attributes
    for column in df:
        diff = max(df[column]) - min(df[column])
        df[column] = (df[column] - min(df[column])) / diff

    df = df.astype(float)

    return df


def load_file(file_name):
    # load data into dataframe
    titles = ['age', 'sex', 'pain_type','blood_pressure', 'serum_cholestoral',
              'blood_sugar', 'electrocardiographic', 'maximum_heart_rate',
              'exercise', 'oldpeak', 'slope_of_peak_exercise', 'number_of_vessels',
              'thal', 'target']
    titles_backup = ['a'+str(i) for i in range(1,15)]

    df = pd.read_csv(file_name, sep=',', header=None, names=titles_backup)

    # clean invalid rows
    df = df.dropna()
    df = df.astype(str)
    for title in titles_backup:
        df.drop(df.index[df[title] == '?'], inplace=True)
    df = df.astype(float)

    df = shuffle(df)
    return df


if __name__ == '__main__':

    # file_name = './kongzhang_preprocessed.csv'
    # df = pd.read_csv(file_name)
    df = load_file('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
    df = pre_process_data_optimised(df)
    # df = pre_process_data(df)
    # train = pd.DataFrame(df.iloc[:190])

    attribute_influence = {}

    for column in df:
        if column == 'a14':
            continue
        train = df.drop([column], axis=1)
        test = df.drop([column], axis=1)
        total = 0
        match = 0
        for row in test.index:
            total += 1
            if predict(test.loc[row][:-1], test, 3) == df.loc[row][-1]:
                match += 1
        result = str(match / total * 100) + "%"
        attribute_influence[column] = result

    for attribute in sorted(attribute_influence, key=lambda x : attribute_influence[x]):
        print("%s,%s" % (attribute, attribute_influence[attribute]))


    train = pd.DataFrame(df.iloc[:])
    test = pd.DataFrame(df.iloc[:])
    result = []
    for k in range(3, 4):
        total = 0
        match = 0
        for row in test.index:
            total += 1
            if predict(test.loc[row][:-1], test, k) == df.loc[row][-1]:
                match += 1
        result.append(match / total)
    print(result)
