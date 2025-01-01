import yaml

from Problems.Problem_One_Titanic.Models.Logistic_Regression import train_logistic_regression
from Problems.Problem_One_Titanic.Models.decision_tree import train_decision_tree
from Problems.Problem_One_Titanic.Models.random_forest import train_random_forest
from Problems.Problem_One_Titanic.Models.svm import train_svm
from Problems.Problem_One_Titanic.Models.naive_bayes import train_naive_bayes
from Problems.Problem_One_Titanic.Models.KNN import train_knn
from Problems.Problem_One_Titanic.Models.Neural_Network import train_neural_network

from Problems.Problem_One_Titanic.utils.Data_loader import  Titanic_Data_Preparation

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load data
X_train,X_val,X_test,y_train,y_val,y_test,features= Titanic_Data_Preparation()


#Notes on the Code to Be Modified
# 1. This Training Procedure is flawed. It doesn't use of the Validation data. So we can't
#       assert the overfit from the under fit. And this is strange!!!
# 2. We need to be able to change the hyperparameter so that we can make a better fit with
#       the cross validation data
# 3. Design Decision Taken : We will run and edit hyperparameter for each method on its file
#       so that we can easily work with it. After iterating we will choose for every algorithm
#       its fine-tuned parameters.Logically, this python will be used to demonstrate
#       and visualize the different results.

solver = "liblinear"
train_logistic_regression(X_train, X_test, y_train, y_test)
train_decision_tree(X_train, X_test, y_train, y_test, max_depth=config["decision_tree"]["max_depth"])
train_random_forest(X_train, X_test, y_train, y_test, n_estimators=config["random_forest"]["n_estimators"])
train_svm(X_train, X_test, y_train, y_test, kernel=config["svm"]["kernel"], C=config["svm"]["C"], gamma=config["svm"]["gamma"])
train_naive_bayes(X_train, X_test, y_train, y_test)
train_knn(X_train, X_test, y_train, y_test, neighbors=config["knn"]["neighbors"])
train_neural_network(X_train, X_test, y_train, y_test, hidden_layer_sizes=tuple(config["neural_network"]["hidden_layers"]), max_iter=config["neural_network"]["max_iter"])



