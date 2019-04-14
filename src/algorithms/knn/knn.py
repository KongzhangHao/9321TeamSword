import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def predict(test_data, training_data, k):
    training_data['distance'] = 0
    temp = (training_data.iloc[:, :-2] - test_data) ** 2
    training_data['distance'] = temp.sum(1) ** 1 / 2
    knn = training_data.sort_values(by='distance')[:k]
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

    # df = shuffle(df)
    return df



def preprocess_input(row, df):
    # normalize continuous value of attributes
    for column in row:
        if column == 'a1' or column == 'a4' or column == 'a5' or column == 'a8' or column == 'a10':
            diff = max(df[column]) - min(df[column])
            row[column] = (row[column] - min(df[column])) / diff

    # split discrete value of attributes
    # attribute 3
    row['a3_1'] = row['a3'].map(lambda x: x == 1)
    row['a3_2'] = row['a3'].map(lambda x: x == 2)
    row['a3_3'] = row['a3'].map(lambda x: x == 3)
    row['a3_4'] = row['a3'].map(lambda x: x == 4)
    row = row.drop(['a3'], axis=1)

    # attribute 7
    row['a7_0'] = row['a7'].map(lambda x: x == 0)
    row['a7_1'] = row['a7'].map(lambda x: x == 1)
    row['a7_2'] = row['a7'].map(lambda x: x == 2)
    row = row.drop(['a7'], axis=1)

    # attribute 11
    row['a11_1'] = row['a11'].map(lambda x: x == 1)
    row['a11_2'] = row['a11'].map(lambda x: x == 2)
    row['a11_3'] = row['a11'].map(lambda x: x == 3)
    row = row.drop(['a11'], axis=1)

    # attribute 12
    row['a12_0'] = row['a12'].map(lambda x: x == 0)
    row['a12_1'] = row['a12'].map(lambda x: x == 1)
    row['a12_2'] = row['a12'].map(lambda x: x == 2)
    row['a12_3'] = row['a12'].map(lambda x: x == 3)
    row = row.drop(['a12'], axis=1)

    # attribute 13
    row['a13_3'] = row['a13'].map(lambda x: x == 3)
    row['a13_6'] = row['a13'].map(lambda x: x == 6)
    row['a13_7'] = row['a13'].map(lambda x: x == 7)
    row = row.drop(['a13'], axis=1)

    row = row.astype(float)

    # re-arrange attributes columns
    row = row[['a1', 'a2', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a4', 'a5', 'a6',
             'a7_0', 'a7_1', 'a7_2', 'a8', 'a9', 'a10', 'a11_1', "a11_2", "a11_3", 'a12_0',
             'a12_1', 'a12_2', 'a12_3', 'a13_3', 'a13_6', 'a13_7']]

    return row


def main():

    # file_name = './kongzhang_preprocessed.csv'
    # df = pd.read_csv(file_name)
    df = load_file('heart_disease.csv')
    df = pre_process_data_optimised(df)
    # df = pre_process_data(df)
    # train = pd.DataFrame(df.iloc[:190])

    test = pd.DataFrame(df.iloc[:])
    result = []
    k = 3
    total = 0
    match = 0
    for row in test.index:
        total += 1
        if predict(test.loc[row][:-1], test, k) == df.loc[row][-1]:
            match += 1
    print("kNN accuracy: %.2f" % (match / total * 100) + "%")
    print("Accuracy of removing different attribute each time")

    attribute_influence = {}

    for column in df:
        if column == 'a14':
            continue
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


if __name__ == '__main__':
    main()