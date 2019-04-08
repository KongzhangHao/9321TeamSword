import pandas as pd
import numpy as np
from sklearn.utils import shuffle



if __name__ == '__main__':

    file_name = './kongzhang_preprocessed.csv'
    #df = pd.read_csv(file_name)
    df = load_file('/Users/hao/PycharmProject/COMP9321-project/data/heart_disease.csv')
    df = pre_process_data_optimised(df)
    df = pre_process_data(df)
    train = pd.DataFrame(df.iloc[:190])
