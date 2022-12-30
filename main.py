import requests
import pandas as pd
import json

if __name__ == '__main__':
    # 1 send request to get test file
    # train_data = pd.read_csv('train_dataset.csv')
    # train_data = train_data.head(2000)
    # train_data_json = train_data.to_json(orient="index")
    # response = requests.post('http://127.0.0.1:8200/get_test_data',json=train_data_json)
    # 2 send request to get clean train data
    # train_data = pd.read_csv('train_dataset.csv')
    # train_data_json = train_data.to_json(orient="index")
    # response = requests.post('http://127.0.0.1:8200/get_train_data')
    # 3 send request to get clean test data with one hot
    test_data = pd.read_csv('test1.csv', index_col=0)
    test_data.drop(columns=['Sale'], inplace=True)
    test_data_json = test_data.to_json(orient="index")
    response = requests.post('http://127.0.0.1:8200/get_test_data_one_hot', json=test_data_json)
    print(response.text)
    # transform json file to dataframe
    # data = json.dumps(response.json(), indent=4)
    # data = pd.read_json(data, orient='index')
    # print(data)
    # data.to_csv('test8585.csv')
    # data = json.loads(response.text)
    # df = pd.json_normalize(data)
