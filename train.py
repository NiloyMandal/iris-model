import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

def main():
    # Load dataset
    df = pd.read_csv('iris.csv')
    
    # Check for missing values and drop them if any
    df = df.dropna()

    # Features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    # Strict Featurization Ordering: split before fitting preprocessing pipelines
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with standard scaler and random forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(pipeline, 'model.pkl')
    print("Model saved to model.pkl")

if __name__ == '__main__':
    main()
