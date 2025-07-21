# mlops-project
This repo showcases MLOps projects- home lab

# Enviroment Setup
The env setup for conda is avaiable in the below repo
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# start the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db

# command to convert notebook to python file
jupyter nbconvert --to=script <filename>
