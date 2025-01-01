from sklearn.neural_network import MLPClassifier
from utils.evaluation import evaluate_model

def train_neural_network(X_train, X_test, y_train, y_test, hidden_layer_sizes=(64, 32), max_iter=1000):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    print("Neural Network Results:")
    evaluate_model(model, X_test, y_test)