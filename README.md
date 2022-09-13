# iris_classification

This notebook is used for creating the model. I modified the input of the tuned model to adjusting the required 3 dimension input for triton. I also changing the scikit learn normalize to normalizer so that the normalize input can be applied to regular inference input.
'''
AI_in_Full_Bloom_Classifying_Iris_Flowers_with_Code.ipynb
'''

## A. Deploy model

Im using [triton inference server](https://github.com/triton-inference-server/) to deploy the model. 
for this case, i saved keras model to .h5 file, then convertin into tensorrt so that the model can be loaded on the server.

this model need to be run on docker.
how to use:
1. pull Triton inverence server using setup.sh.
2. run the server with the saved model on model repository by modifiying the model_repository path on run_inference_server.sh.
3. docker container will run on your device.

## B. Model inference scheduler
I'm using simple python scheduler that will run on docker. [triton inference client](https://raw.githubusercontent.com/triton-inference-server/client) is used for communicate with deployed model.

how to use:
1. build the docker images using build.sh
2. init mongodb database using init_mongo_db.sh
3. use add initial data.ipynb to add dummy initial data. 
4. run the docker image on container by executing run_service.sh
5. the scheduler will be running

## C. Model API
I'm using [tornado web server](https://github.com/tornadoweb/tornado) library to run simple web api server for this task.

how to run the service:
1. build the docker images using build.sh
2. run the docker image on container by executing run_service.sh
3. the api service will be run on local device on port 9999
4. use postman collection on the folder to test the API.
