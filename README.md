# Breast Cancer Prediction App

A Streamlit-based web application for predicting breast cancer diagnosis using machine learning. This application uses a Logistic Regression model trained on the Wisconsin Breast Cancer dataset to predict whether a tumor is malignant (M) or benign (B) based on various features.

## Features

- Interactive web interface built with Streamlit
- Machine learning model for breast cancer prediction
- Data visualization capabilities
- User-friendly input form for prediction
- Model performance metrics and insights

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd streamlit-cancer-predict
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
streamlit-cancer-predict/
├── app/
│   └── main.py          # Streamlit application
├── model/
│   └── main.py          # Model training and preprocessing
├── data/
│   └── data.csv         # Dataset
├── assets/              # Static assets
└── requirements.txt     # Project dependencies
```

## Usage

1. Train the model:
```bash
cd model
python main.py
```

2. Run the Streamlit app:
```bash
cd app
streamlit run main.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Model Details

The application uses a Logistic Regression model with the following features:
- Standardized input features
- 80-20 train-test split
- StandardScaler for feature normalization

## Dependencies

- numpy==1.23.4
- pandas==1.5.1
- plotly==5.11.0
- scikit_learn==1.2.2
- streamlit
- altair<5.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wisconsin Breast Cancer Dataset
- Streamlit for the web framework
- Scikit-learn for machine learning capabilities
