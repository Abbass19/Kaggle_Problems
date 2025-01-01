from Data_loader import Titanic_Data_Preparation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#Use the Custom Function for Titanic Data
X_train,X_val,X_test,y_train,y_val,y_test = Titanic_Data_Preparation()
print(f"The shape of X_train is {X_train.shape}")

#Choose a binary classification Model : We can Test all the models we know
#The process is that we train and fit. And

# 1.Logistic Regression
logistic_model_1 = LogisticRegression(max_iter=1000)
logistic_model_1.fit(X_train, y_train)
y_val_pred = logistic_model_1.predict(X_val)

# Evaluate model accuracy on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Make predictions on the test set
y_test_pred = logistic_model_1.predict(X_test)

# Evaluate model accuracy on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")


# 2.Decision Tree
# 3.Random Forest
# 4.Support Vector Machine
# 5.Naive Bayes
# 6.KNN
# 7.Neural Network




