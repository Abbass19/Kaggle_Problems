from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    # print(f"Accuracy: {accuracy:.4f}")
    # print("Classification Report:")
    # print(classification_report(y_test, predictions))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, predictions))
    return accuracy
