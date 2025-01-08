import numpy as np
import yaml
import os

from sklearn.linear_model import LogisticRegression

from Problems.Problem_One_Titanic.utils.Data_loader import Titanic_Data_Preparation
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model
def train_logistic_regression(X_train, X_test, y_train, y_test, penalty="l2"):
    model = LogisticRegression(penalty=penalty, solver="liblinear", max_iter=200 , C=1)
    model.fit(X_train, y_train)
    print(f"Logistic Regression (penalty={penalty}) Results:")
    evaluate_model(model, X_test, y_test)




#Working with Directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
parent_dir = os.path.dirname(current_dir)  # Go one level up
config_path = os.path.join(parent_dir, "config.yaml")

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load data
X_train,X_val,X_test,y_train,y_val,y_test,features= Titanic_Data_Preparation()

#Testing and Fine-Tuning will start here

# Initialize hyperparameter ranges
penalties = ["l1", "l2"]
C_values = [0.1, 1, 10, 100]
solvers = ['liblinear' , 'lbfgs']
max_iterations = [200, 500, 800, 1100, 1400]
results = {}

#Variables for Finding Optimized parameter
best_model = None
best_accuracy = 0
best_hyperparameters = {}
hyperparameters  = np.zeros((0,5))
worst_accuracy = 0.78
mean_accuracy = 0

#This is a game only for printing :
#These are 80 lines
# for penalty in penalties:
#     for C_value in C_values:
#         for solver in solvers:
#             if solver == 'lbfgs' and penalty == 'l1':
#                 continue
#             for iteration in max_iterations:
#                 print(f"Training model with penalty : {penalty} C_value: {C_value} {solver} iterations: {iteration} ")
#



for penalty in penalties:
    for C_value in C_values:
        for solver in solvers:
            if solver== 'lbfgs' and penalty== 'l1':
                continue
            for iteration in max_iterations:
                print(f"Training model with {penalty} {C_value} {solver} {iteration} iterations")

                model = LogisticRegression(penalty=penalty,C=C_value ,solver=solver,max_iter=iteration)
                model.fit(X_train, y_train)
                train_accuracy = evaluate_model(model, X_train, y_train)
                cv_accuracy = evaluate_model(model, X_val, y_val)
                test_accuracy = evaluate_model(model,X_test,y_test)
                new_record = np.array([penalty,C_value,solver,iteration,test_accuracy])
                hyperparameters =  np.vstack([hyperparameters,new_record])

                # print(f"    Training accuracy           : {train_accuracy}")
                # print(f"    Cross-Validation accuracy   : {cv_accuracy}")
                # print(f"    Testing accuracy            : {test_accuracy}")

                # Store results in the dictionary
                key = (penalty, C_value, solver, iteration)
                results[key] = {
                    'training error': train_accuracy,
                    'cv_accuracy': cv_accuracy,
                    'test_accuracy': test_accuracy
                }

                #Calculating the mean precision
                mean_accuracy += test_accuracy

                if test_accuracy>best_accuracy:
                    best_accuracy = test_accuracy
                    best_hyperparameters = {penalty,C_value,solver,iteration}
                    best_model=model

                if test_accuracy < worst_accuracy != 0:
                    worst_accuracy=test_accuracy

print(f"The winner of the Logistic Regression Problem is:")
print(f"\n {best_hyperparameters}")
print(f"\nHaving a testing error of {best_accuracy}")
print(f"Overpassing the mean of models that is {mean_accuracy/60}")
print(f"The Worst is {worst_accuracy}")



#We need to analyze the dictionary we have to build the insight of
#how to mitigate hyperparameter to make testing error less

#Make sure the numpy array is Working :
print(f"The shape of hyperparameter array is {hyperparameters.shape}")
print(f"Printing the hyperparameter Array : ")
print(f"{hyperparameters}")

#Node --> More accuracy (Feature 1 ,Feature 2 , Feature 3, Feature 4)
# What is the direction for Titanic Logistic Regression to have higher accuracy

#This question introduces the talk on Grid Search or Random Search. This is something you can explore
#tomorrow maybe.






