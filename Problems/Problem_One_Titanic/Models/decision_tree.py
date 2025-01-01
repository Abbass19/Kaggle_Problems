from sklearn.tree import DecisionTreeClassifier
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model

def train_decision_tree(X_train, X_test, y_train, y_test, max_depth=5):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    print("Decision Tree Results:")
    evaluate_model(model, X_test, y_test)

#What is the procedure for Decision Tree