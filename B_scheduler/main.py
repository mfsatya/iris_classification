import schedule
import time
import pymongo
import random
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing

def test_infer(model_name,
               input_data,
               headers=None,
               request_compression_algorithm=None,
               response_compression_algorithm=None):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('input_1', [1, 1, 1, 4], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input_data)

    outputs.append(httpclient.InferRequestedOutput('dense_6/Softmax'))
    query_params = {'test_1': 1}
    
    triton_client = httpclient.InferenceServerClient(
                url="localhost:8000",
                verbose=1)
    
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)
    return results



client = pymongo.MongoClient("127.0.0.1", 27017)
db = client.iris_database
iris_input = db.iris_input
iris_output = db.iris_output
input_data = list(iris_input.find({}))
today = date.today()

iris_dataset=pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
X = iris_dataset.iloc[:, 0:4].values
scaler = preprocessing.Normalizer().fit(X)

def job():
    for i in range(10): #insert random data for inference every 10AM
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sepal_length = random.uniform(1.0, 8.0)
        sepal_width = random.uniform(2.0, 4.5)
        petal_length = random.uniform(2.0, 7.0)
        petal_width = random.uniform(0.0, 2.5)
        data = {
            'time': time,
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        iris_input.insert_one(data)
    
    
    output_pred = {}
    for data in list(iris_input.find({})):
        date = datetime.strptime(data['time'], '%Y-%m-%d %H:%M:%S').date()
        if (date == today):
            input_data = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
            input_data = scaler.transform([input_data])
            input_data = np.array(input_data).astype(np.float32)
            input_data = np.reshape(input_data, (1, 1, 1, 4))
            result = test_infer(model_name="iris_classification", input_data = input_data)
            result = result.as_numpy('dense_6/Softmax')
            result = np.squeeze(result)
            output_pred[data['_id']] = np.argmax(result)

    for pred_id in output_pred.keys():
        data = {
            'id' : pred_id,
            'class': int(output_pred[pred_id])
        }
        iris_output.insert_one(data)
schedule.every().day.at("10:00").do(job) #scheduler

while True:
    schedule.run_pending()
    time.sleep(1)