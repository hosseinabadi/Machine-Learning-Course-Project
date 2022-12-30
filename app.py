from flask import Flask, request
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import datetime
import json
from flask_jsonpify import jsonpify
import requests
# sns.set()
# plt.style.use("ggplot")

import warnings

warnings.filterwarnings('ignore')
import random

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


def replace_minus_one_with_nan(df):
    num_cols = df._get_numeric_data().columns
    categ_cols = set(df.columns) - set(num_cols)
    for column in num_cols:
        df[column][df[column] == -1] = np.nan
    for column in categ_cols:
        df[column][df[column].isin(["-1"])] = np.nan
    return df


def time_stamp_to_day_and_hour(df):
    df['click_timestamp'] = df['click_timestamp'].astype(str)
    df['click_timestamp'] = df['click_timestamp'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
    # df['day'] = df['click_timestamp'].apply(lambda d: d.day)
    df['hour'] = df['click_timestamp'].apply(lambda d: d.hour)
    df = df.drop(columns=['click_timestamp'])
    return df


def impute_data(mode, data, num_cols, categ_cols):
    if mode == 'mode':
        # fill values by mode
        for column in data.columns:
            data[column].fillna(data[column].mode()[0], inplace=True)
    elif mode == 'mode_mean':
        # categs with mode and numerics with mean
        for column in categ_cols:
            data[column].fillna(data[column].mode()[0], inplace=True)
        for column in num_cols:
            data[column].fillna(data[column].mean(), inplace=True)
    elif mode == 'distribution':
        # fill columns with distribution of the column
        for column in data.columns:
            s = data[column].value_counts(normalize=True)
            missing = data[column].isnull()
            data.loc[missing, column] = np.random.choice(s.index, size=len(data[missing]), p=s.values)
    elif mode == 'replace_with_constant_values':
        # set a new value for nan values , categorical and numerical
        for column in categ_cols:
            data[column].fillna("Unknown", inplace=True)
        for column in num_cols:
            data[column].fillna(-1000, inplace=True)
    return data


def read_and_clean_data(data):
    data['product_price'].replace({0: -1}, inplace=True)
    data = replace_minus_one_with_nan(data)
    data = time_stamp_to_day_and_hour(data)
    data = data.drop(
        columns=['SalesAmountInEuro', 'user_id', 'time_delay_for_conversion', 'product_category(7)',
                 'product_category(5)',
                 'product_category(6)'])
    # column that should be removen because of so many categories
    data = data.drop(columns=['product_id', 'product_title', 'product_brand', 'audience_id', 'product_price'])
    # data = remove_outliers(data)
    return data


def remove_outliers(data):
    data['nb_clicks_1week'][data['nb_clicks_1week'] > 15000] = 10000
    return data


def impute_train_data(data):
    num_cols = data._get_numeric_data().columns
    categ_cols = set(data.columns) - set(num_cols)
    drop_index = data[(data.isnull().sum(axis=1) >= 2) & (data['Sale'] == 0)].index
    data_new = data.drop(data.index[drop_index])
    impute_data('distribution', data_new, num_cols, categ_cols)
    return data_new


@app.route('/get_train_data', methods=["POST"])
def get_final_train_data():
    df = pd.read_csv('./train_dataset.csv')
    df = read_and_clean_data(df)
    one_hot_encoding(df, test=None)

    df = impute_train_data(df)
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    return parsed


@app.route('/get_test_data', methods=["POST"])
def get_final_test_data():
    test_df = request.json
    test_df = pd.read_json(test_df, orient='index')
    test_df = read_and_clean_data(test_df)
    train_df = pd.read_csv('./train_dataset.csv')
    train_df = read_and_clean_data(train_df)
    train_df = impute_train_data(train_df)
    num_cols = train_df._get_numeric_data().columns
    categ_cols = set(train_df.columns) - set(num_cols)
    test_length = len(test_df)
    final_df = train_df.append(test_df)
    impute_data('distribution', final_df, num_cols, categ_cols)
    final_test_df = final_df.tail(test_length)
    result = final_test_df.to_json(orient="index")
    parsed = json.loads(result)
    return parsed


@app.route('/get_test_data_one_hot', methods=["POST"])
def get_test_data_one_hot():
    test_df = request.json
    test_df = pd.read_json(test_df, orient='index')
    test_df = read_and_clean_data(test_df)
    train_df = pd.read_csv('./train_dataset.csv')
    train_df = read_and_clean_data(train_df)
    train_df = impute_train_data(train_df)
    num_cols = train_df._get_numeric_data().columns
    categ_cols = set(train_df.columns) - set(num_cols)
    test_length = len(test_df)
    final_df = train_df.append(test_df)
    impute_data('distribution', final_df, num_cols, categ_cols)
    test_df = final_df.tail(test_length)
    final_test_df = one_hot_encoding(train_df, test_df)
    length = len(final_test_df.columns)
    if 256 > length:
        for i in range(256 - length):
            final_test_df[f'{i}'] = 0
    else:
        final_test_df = final_test_df.iloc[:, :-(length-256)]

    result = final_test_df.to_json(orient="records")
    res =  requests.post('XGBoost:8080/invocations', data=result,headers={"Content-Type": "application/json; format=pandas-records"})
    return res.text,res.status_code,res.headers.items()
    # parsed = json.loads(result)
    # return parsed


def one_hot_encoding(train, test):
    cut_value = [150, 10, 80, 100, 100,
                 100, 60, 100, 50]
    num_cols = train._get_numeric_data().columns
    categ_cols = set(train.columns) - set(num_cols)
    for i, column_name in enumerate(categ_cols):
        #         column_name = f'product_category({i})'
        tmp1 = train[column_name].value_counts()
        tmp2 = tmp1[tmp1 > cut_value[i]].index.tolist()
        df_t = (~test[column_name].isin(tmp2)).astype(int).to_frame()
        print(column_name, len(tmp2))
        for j, item1 in enumerate(tmp2):
         # print(item1)
         # print(df[column_name].isin([item1]).astype(int))
            df_t = pd.concat([df_t, test[column_name].isin([item1]).astype(int).rename(column_name + f"({j})")], axis=1)
        #
        test = test.drop(columns=column_name)
        test = pd.concat([test, df_t], axis=1)
    return test


if __name__ == '__main__':
    app.run(port=8200)
