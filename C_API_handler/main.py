import asyncio
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
import tornado.web
        
iris_dataset=pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
X = iris_dataset.iloc[:, 0:4].values
scaler = preprocessing.Normalizer().fit(X)

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
                url="172.17.0.1:8000",
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

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Welcome to our inferce API")
        
class InferenceOnehandler(tornado.web.RequestHandler):
    async def get(self):
        sepal_length = self.get_argument("sepal_length", default="")
        sepal_width = self.get_argument("sepal_width", default="")
        petal_length = self.get_argument("petal_length", default="")
        petal_width = self.get_argument("petal_width", default="")
        result = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        
        input_data = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
        input_data_scaled = scaler.transform([input_data])
        input_data = np.array(input_data_scaled).astype(np.float32)
        input_data = np.reshape(input_data, (1, 1, 1, 4))
        result = test_infer(model_name="iris_classification", input_data = input_data)
        result = result.as_numpy('dense_6/Softmax')
        result = np.squeeze(result)
        
        self.write({"code": 200, "inference results": str(np.argmax(result))})

    async def post(self):
        content_type = self.request.headers.get("Content-Type")
        if content_type == "application/json":
            try:
                body = json.loads(self.request.body.decode('utf-8'))
            except json.decoder.JSONDecodeError:
                self.write(
                    {"code": 1401, "results": BASE_HANDLER_ERRORS[1401]})
                self.set_status(400)
                return
            sepal_length = body.get("sepal_length", "")
            sepal_width = body.get("sepal_width", "")
            petal_length = body.get("petal_length", "")
            petal_width = body.get("petal_width", "")

        else:
            sepal_length = self.get_body_argument("sepal_length", default="")
            sepal_width = self.get_body_argument("sepal_width", default="")
            petal_length = self.get_body_argument("petal_length", default="")
            petal_width = self.get_body_argument("petal_width", default="")
            
        result = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        
        # iris_dataset=pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
        # X = iris_dataset.iloc[:, 0:4].values
        # scaler = preprocessing.Normalizer().fit(X)
        
        input_data = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
        input_data_scaled = scaler.transform([input_data])
        input_data = np.array(input_data_scaled).astype(np.float32)
        input_data = np.reshape(input_data, (1, 1, 1, 4))
        result = test_infer(model_name="iris_classification", input_data = input_data)
        result = result.as_numpy('dense_6/Softmax')
        result = np.squeeze(result)
        
        self.write({"code": 200, "inference results": str(np.argmax(result))})

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/inference_one", InferenceOnehandler),
        (r"/inference_many", InferenceOnehandler),
    ])


async def main():
    app = make_app()
    app.listen(9999)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
   