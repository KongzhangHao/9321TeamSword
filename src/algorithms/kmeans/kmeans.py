# This is written by Shaowen Tang for COMP9321 Assignment3
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans





def load_file(file_name):
    titles = ['age', 'sex', 'pain_type','blood_pressure', 'serum_cholestoral',
              'blood_sugar', 'electrocardiographic', 'maximum_heart_rate',
              'exercise', 'oldpeak', 'slope_of_peak_exercise', 'number_of_vessels',
              'thal', 'target']
    titles_backup = ['a'+str(i) for i in range(1,15)]

    df = pd.read_csv(file_name, sep=',', header=None, names=titles_backup)

    df = df.dropna()
    df = df.astype(str)
    for title in titles_backup:
        df.drop(df.index[df[title] == '?'], inplace=True)
    df = df.astype(float)

    df = shuffle(df)
    return df


def pre_process_data(df):
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

def cluster_by_kmeans(df):
    df_without_label = df.drop(['a14'], axis=1)

    estimator = KMeans(n_clusters=2)
    estimator.fit(df_without_label)

    labels = estimator.labels_
    facts = np.array(df['a14'])

    count_error = 0
    for index in range(len(labels)):
        if facts[index] != labels[index]:
            count_error += 1

    print('Accuracy: %.2f' % (max(count_error, len(facts) - count_error)/len(facts)*100))
    return (max(count_error, len(facts) - count_error)/len(facts)*100)


if __name__ == '__main__':
    df = load_file('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
    df_processed = pre_process_data(df)
    print_dataframe(df_processed)

    cluster_by_kmeans(df_processed)
    attribute_influence = {}

    for column in df_processed:
        if column == 'a14':
            continue
        attribute_influence[column] = cluster_by_kmeans(df_processed.drop([column], axis=1))

    for attribute in sorted(attribute_influence, key=lambda x : attribute_influence[x]):
        print("%s,%s" % (attribute, attribute_influence[attribute]))
