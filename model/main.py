
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    # Drop the 'diagnosis' column from the feature set
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Save column names before scaling
    feature_columns = X.columns.tolist()

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return model, scaler, feature_columns

def get_clean_data():
    data = pd.read_csv("data/data.csv")

    # Drop unnecessary columns including specific columns
    columns_to_drop = [
        'Unnamed: 32', 'id', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
        'concavity_worst', 'concave points_worst', 'symmetry_worst',
        'fractal_dimension_worst'
    ]
    data = data.drop(columns=columns_to_drop, axis=1)

    # Map 'diagnosis' to binary values
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def main():
    data = get_clean_data()
    model, scaler, feature_columns = create_model(data)

    # Save model, scaler, and feature columns using pickle
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('model/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)

if __name__ == '__main__':
    main()
