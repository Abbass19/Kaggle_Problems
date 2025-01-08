from sklearn.neighbors import KNeighborsClassifier
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model

def train_knn(X_train, X_test, y_train, y_test, neighbors=5):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X_train, y_train)
    print("K-Nearest Neighbors Results:")
    evaluate_model(model, X_test, y_test)


