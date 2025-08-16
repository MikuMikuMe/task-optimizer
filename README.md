# Task-Optimizer

Creating a task optimizer using machine learning involves several steps, including data collection, preprocessing, model selection, and prediction. For simplicity, we'll focus on a basic version that utilizes a machine learning model to prioritize tasks based on given features such as deadlines and importance. This example will use a simple decision tree classifier. Note that comprehensive user behavior analytics would require more advanced techniques and data collection, likely involving a more complex machine learning pipeline.

Below is a Python program that demonstrates a simplified version of this concept:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Set up logging for debugging and error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load or create sample data for task prioritization.
    This function creates a simple dataset with task attributes.

    Returns:
        pd.DataFrame: DataFrame containing task attributes and priority.
    """
    # Sample data: In a real-world scenario, this data would come from a database or user input
    data = {
        'deadline_days': [1, 3, 2, 5, 4],
        'importance_level': [5, 3, 4, 1, 2],
        'priority': [1, 0, 1, 0, 0]  # 1: High Priority, 0: Low Priority
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Preprocess the dataset into features and labels.

    Args:
        df (pd.DataFrame): DataFrame containing task data.

    Returns:
        tuple: Features (X) and labels (y) for the model.
    """
    X = df[['deadline_days', 'importance_level']]
    y = df['priority']
    return X, y

def train_model(X, y):
    """
    Train a decision tree classifier on the given data.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        DecisionTreeClassifier: Trained Decision Tree model.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model training complete. Accuracy: {accuracy:.2f}")
        return clf
    except Exception as e:
        logging.error("Error during model training", exc_info=True)

def predict_priority(clf, task_features):
    """
    Predict the priority of a new task.

    Args:
        clf (DecisionTreeClassifier): Trained classifier.
        task_features (list): Feature list of the new task.

    Returns:
        int: Predicted priority (1: High, 0: Low).
    """
    try:
        return clf.predict([task_features])[0]
    except Exception as e:
        logging.error("Error in predicting task priority", exc_info=True)

def main():
    try:
        # Load and preprocess data
        df = load_data()
        X, y = preprocess_data(df)

        # Train model
        clf = train_model(X, y)

        # Example: Predict the priority of a new task
        new_task = [2, 4]  # Example features: 2 days deadline, importance level 4
        predicted_priority = predict_priority(clf, new_task)
        logging.info(f"Predicted priority for the new task: {'High' if predicted_priority == 1 else 'Low'}")
    except Exception as e:
        logging.critical("Critical error in the main function", exc_info=True)

if __name__ == "__main__":
    main()
```

### Explanation
1. **Data Loading**: Creates a sample dataset mimicking task attributes and priorities.
2. **Preprocessing**: Splits the dataset into features and labels.
3. **Model Training**: Uses a Decision Tree Classifier to train the model on the tasks.
4. **Prediction**: Demonstrates how to predict the priority of a new task.
5. **Logging and Error Handling**: Uses Python's `logging` library to handle and log errors, providing informative messages and stack traces if things go wrong.

### Notes
- This example uses a small, static dataset for demonstration. In practice, you'd gather data dynamically from user interactions.
- Model accuracy can be improved by tuning hyperparameters or using more advanced models.
- Implementing comprehensive user behavior analytics would involve collecting and processing user-specific data, which adds complexity not covered here.
