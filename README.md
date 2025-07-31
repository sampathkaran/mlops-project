# mlops-project
This repo showcases MLOps projects- home lab

# Enviroment Setup
The env setup for conda is avaiable in the below repo
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# start the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db

# command to convert notebook to python file
jupyter nbconvert --to=script <filename>

# Prefect repo link for setup
https://github.com/discdiver/prefect-mlops-zoomcamp.git

$ conda create -n prefect-env python=3.9.12
$ pip install -r requirements.txt 
$ prefect server start