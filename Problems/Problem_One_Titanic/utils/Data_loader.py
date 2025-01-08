import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
def read_csv(filename):
    with open(filename) as file:
        header = file.readline().strip().split(',')[1:]
        data = []
        for line in file:
            row = line.strip().split(',')[1:]
            data.append(row)

    return np.array(data), header

def restructure_data(data):
    selected_features = [1,3,4,5,6,8,10]
    Y = data[:, 0]
    X = data[:, selected_features]
    return X,Y

def remove_rows_with_missing_values(X, Y):
    # Find rows with missing values in X
    rows_with_missing_X = np.any(np.isnan(X), axis=1)

    # Keep rows without missing values in X
    X_clean = X[~rows_with_missing_X]

    # Keep corresponding rows in Y
    Y_clean = Y[~rows_with_missing_X]

    return X_clean, Y_clean

def encode_feature(data):
    encoded_data = np.zeros_like(data, dtype=int)  # Ensure integer dtype
    encoded_data[data == 'S'] = 0
    encoded_data[data == 'Q'] = 1
    encoded_data[data == 'C'] = 2
    return encoded_data

def Titanic_Data_Preparation(printing = False):
    """This is a procedure to Make the Data of the Titanic ready to
        implement with different binary classification models. Not something to Generalize """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, "train.csv")

    # filename = "train.csv"
    data, header = read_csv(filename)
    X, Y = restructure_data(data)
    # Change feature 1 from text (female - male) to number (0 - 1)
    X[:, 1] = (X[:, 1] == 'male').astype(int)

    # Change feature 6 from C,Q,S
    X[:, 6] = encode_feature(X[:, 6])

    # Change feature of index 2 from text to number
    for i in range(X.shape[0]):
        try:
            X[i, 2] = int(X[i, 2])
        except (ValueError, IndexError):
            X[i, 2] = np.nan

    # We need to change all features from <U82 to number so that we remove empty values ...
    X = X.astype(np.float64)

    # Removing Empty
    X, Y = remove_rows_with_missing_values(X, Y)
    # Normalizing the continuous Data (Age and Fair Only) to ease Convergence :
    scaler = StandardScaler()
    continuous_features_indices = [2, 5]
    X[:, continuous_features_indices] = scaler.fit_transform(X[:, continuous_features_indices])
    features = ["P_class", "Sex", "Age", "SibSP", "Parch", "Fare", "Embarked"]
    X_train, X_temp, y_train, y_temp = train_test_split(X,Y,test_size=0.3,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)
    if printing:
        print("Training set size:", len(X_train))
        print("Validation set size:", len(X_val))
        print("Test set size:", len(X_test))

    return X_train,X_val,X_test,y_train,y_val,y_test,features