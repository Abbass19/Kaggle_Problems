from sklearn.svm import SVC
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model

def train_svm(X_train, X_test, y_train, y_test, kernel="rbf", C=1.0, gamma="scale"):
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    print("Support Vector Machine Results:")
    evaluate_model(model, X_test, y_test)