from sklearn.ensemble import RandomForestClassifier
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model

def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest Results:")
    evaluate_model(model, X_test, y_test)