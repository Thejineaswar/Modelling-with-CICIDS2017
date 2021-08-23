import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from .getLists import *

PATHS,COL = getList()

WORK_DIR = '' #Set the Directory where you download the CSV files
TRAIN_FOLD = 5



def read_and_clean_df():
    #Reading and concatinating the
    df = pd.read_csv(WORK_DIR + PATHS[0])
    for i in range(1, len(PATHS)):
        temp = pd.read_csv(WORK_DIR + PATHS[i])
        df = pd.concat([df, temp])

    # Handling Null Values
    m = df.loc[df[' Flow Packets/s'] != np.inf,' Flow Packets/s'].max()
    df[' Flow Packets/s'].replace(np.inf,m,inplace=True)
    m = df.loc[df['Flow Bytes/s'] != np.inf,'Flow Bytes/s'].max()
    df['Flow Bytes/s'].replace(np.inf,m,inplace=True)


    null_values = df.isna().sum()
    null_values[null_values >0]
    null_index = np.where(df['Flow Bytes/s'].isnull())[0]
    df.dropna(inplace = True)
    return df

def downsample_and_remove_null(df):
    temp = df[df[' Label'] == 'BENIGN']
    temp[' Destination Port'].describe()
    temp = temp.sample(frac = 0.1,random_state = 42)

    df = df[df[' Label'] != 'BENIGN']
    df = pd.concat([df,temp])
    return df


if __name__ == '__main__':
    df = read_and_clean_df()
    df = downsample_and_remove_null()
    df['folds'] = 0
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for i, (_, test_index) in enumerate(skf.split(df[[' Destination Port']], df[' Label'])):
        df.iloc[test_index, -1] = i

    train_df = df[df['folds'] != 5]
    valid_df = df[df['folds'] == 5]

    scaler = MinMaxScaler()
    train_df[COL] = scaler.fit_transform(train_df[COL])
    valid_df[COL] = scaler.transform(valid_df[COL])

    train_df.to_csv('train_df.csv',index = False)
    valid_df.to_csv('valid_df.csv',index = False)