from sklearn.naive_bayes import GaussianNB
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model

def train_naive_bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Naive Bayes Results:")
    evaluate_model(model, X_test, y_test)
