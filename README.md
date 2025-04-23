# install requirements

pip install -r requirements.txt

# train models

## run all cells in train_cnn_model.ipynb
## run all cells in transfert_learning.ipynb

# load tensorboard

tensorboard --logdir=logs --port=6006

# load mlflow

mlflow ui

# serve the cnn model (change the loaded model if you wanna try the dense one)

streamlit run streamlit_app/app.py
