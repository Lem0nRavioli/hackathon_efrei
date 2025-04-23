# install requirements

pip install -r requirements.txt

# train models

python models/train_dense.py

python models/train_cnn.py

# load tensorboard

tensorboard --logdir=logs --port=6006

# load mlflow

mlflow ui

# serve the cnn model (change the loaded model if you wanna try the dense one)

streamlit run streamlit_app/app.py
