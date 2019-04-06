import pandas as pd
import numpy as np
from sklearn.utils import shuffle


# load file into dataframe and
# clean invalid data
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
    df = load_file('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
    df_processed = pre_process_data(df)
    # df_processed = df
    print_dataframe(df_processed)

