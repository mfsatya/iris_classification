import argparse
import numpy as np
import sys
import gevent.ssl

#Import required libraries 
import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from keras.utils import np_utils

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
    # print(outputs)

    # print(results)
    return results

#  Load the dataset, which contains the data points(sepal length, petal length, etc) and corresponding labels(type of iris)
iris_dataset=pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

iris_dataset.loc[iris_dataset["species"]=="setosa","species"]=0
iris_dataset.loc[iris_dataset["species"]=="versicolor","species"]=1
iris_dataset.loc[iris_dataset["species"]=="virginica","species"]=2

# #This is a debug statement to make sure we uploaded the dataset correctly. 
# #We can comment it out when we actually run the code.
# #print(iris_dataset)

# Break the dataset up into the examples (X) and their labels (y)
X = iris_dataset.iloc[:, 0:4].values
y = iris_dataset.iloc[:, 4].values
X=normalize(X,axis=0)

# Split up the X and y datasets randomly into train and test sets
# 20% of the dataset will be used for the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=31)

#Change the label to one hot vector
'''
[0]--->[1 0 0]
[1]--->[0 1 0]
[2]--->[0 0 1]
'''
y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)

data = X_test[5].astype(np.float32)
input_ = np.reshape(data, (1, 1, 1, 4))
result = test_infer(model_name="iris_classification", input_data = input_)

from keras.models import load_model
model = load_model('./saved_model/h5/model.h5')
model.summary()
prediction=model.predict(input_)

print("Output triton inference server:")
print(result.as_numpy('dense_6/Softmax'))
print("Output keras model:")
print(prediction)