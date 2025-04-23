# Cataract Detection using Deep Learning

This project implements a deep learning solution for detecting cataracts in eye images. It uses a combination of CNN and transfer learning approaches to classify eye images as either normal or showing signs of cataract.

## Project Structure

```
.
├── data/                  # Dataset directory
├── notebooks/            # Jupyter notebooks for data exploration and model training
│   └── cataract_data_exploration.ipynb
├── streamlit_app/        # Web application for model deployment
├── train_cnn_model.ipynb # CNN model training notebook
├── transfert_learning.ipynb # Transfer learning model training notebook
├── best_model.h5         # Trained model weights
└── requirements.txt      # Project dependencies
```

## Features

- Data exploration and preprocessing of cataract image dataset
- Implementation of CNN model for cataract detection
- Transfer learning approach using pre-trained models
- Model training and evaluation
- Web application for real-time cataract detection
- Model performance tracking using MLflow
- Training visualization using TensorBoard

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Exploration
1. Open and run the data exploration notebook:
```bash
jupyter notebook notebooks/cataract_data_exploration.ipynb
```

### Model Training
1. Train the CNN model:
```bash
jupyter notebook train_cnn_model.ipynb
```

2. Train the transfer learning model:
```bash
jupyter notebook transfert_learning.ipynb
```

### Model Monitoring
1. Launch TensorBoard to monitor training:
```bash
tensorboard --logdir=logs --port=6006
```

2. Launch MLflow to track experiments:
```bash
mlflow ui
```

### Web Application
1. Run the Streamlit application:
```bash
streamlit run streamlit_app/app.py
```

## Model Architecture

The project implements two different approaches:
1. Custom CNN model
2. Transfer learning using pre-trained models

Both approaches are designed to classify eye images as either normal or showing signs of cataract.

## Dataset

The project uses the Cataract Image Dataset, which contains:
- Normal eye images
- Cataract-affected eye images
- Split into training and testing sets

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Cataract Image Dataset](https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset)
- TensorFlow and Keras for deep learning implementation
- Streamlit for web application deployment
