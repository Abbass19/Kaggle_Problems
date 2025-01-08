import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC

from Problems.Problem_One_Titanic.utils.Data_loader import Titanic_Data_Preparation
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model

X_train, X_val, X_test, y_train, y_val, y_test, features = Titanic_Data_Preparation()
def check_model_state(training_accuracy, testing_accuracy, baseline=None, tolerance=0.1 ,printing = False):
    """

    Determines the state of a model and calculates the percentage shift of testing error
    relative to training error, printing the status based on the calculated metric.

    Parameters:
        training_accuracy (float): The training error.
        testing_accuracy (float): The testing error.
        baseline (float, optional): The baseline error. Defaults to None.
        tolerance (float, optional): The tolerance for error comparison. Defaults to 0.1.

    Returns:
        float: The metric (difference between testing and training error).
        :param training_accuracy:
        :param testing_accuracy:
        :param baseline:
        :param tolerance:
        :param printing:
    """
    if training_accuracy == 0:
        raise ValueError("Training error cannot be zero to compute percentage shift.")

    # Calculate metric and percentage shift
    reading =  training_accuracy-testing_accuracy
    percentage_shift = (reading / training_accuracy)

    # Determine the state based on metric
    if baseline is not None:
        if abs(training_accuracy - baseline) <= tolerance:
            if abs(percentage_shift) <= tolerance:
                state = "Good"
            else:
                state = "Over-fitting"
        else:
            state = "Under-fitting"
    else:
        if abs(percentage_shift) <= tolerance:
            state = "Uncertain"
        else:
            state = "Over-fitting"

    # Print the status based on the state
    if printing :
        print(f"Status: {state}, Reading: {reading} Shifted by {percentage_shift*100}%")
    return percentage_shift

def train_svm(X_train, X_test, y_train, y_test, kernel="rbf", C=1.0, gamma="scale"):
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    print("Support Vector Machine Results:")
    evaluate_model(model, X_test, y_test)

#This is a promising start :
# Support Vector Machine Results:
# Status: Uncertain, Reading: -0.013645068624321688 Shifted by -1.6777864992150646%
# Having accuracy of 0.8269230769230769


#Function signature is : class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
# shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
# decision_function_shape='ovr', break_ties=False, random_state=None)

#Available kernels kernels = ['linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]
# Hyperparameters we want to change are the following : C , tol ,cache_size, max_iter
#
#
#


#We have 2 generic procedures :




def find_best_hyperparameters(kernel, skip_C = False, skip_Tolerance = False, skip_cache_sizes = False, skip_max_iters = False, degree = 3,printing = False):
    #Some printing for Understanding:
    if printing:
        print(f"Finding Optimal Parameters of kernel : {kernel}")

    # Suppress the ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Define the kernels and hyperparameter ranges
    C_values = [1,0.01, 0.1,  10, 100]
    tol_values = [0.001,0.0001,  0.01,0.1]
    cache_sizes = [200,50 ,100,  500, 1000]
    max_iters = [ -1,500, 1000, 5000, 10000]

    if skip_C:
        C_values = C_values[:1]
    if skip_Tolerance:
        tol_values = tol_values[:1]
    if skip_max_iters:
        max_iters = max_iters[:1]
    if skip_cache_sizes:
        cache_sizes = cache_sizes[:1]

    #Get the Data :
    X_train, X_val, X_test, y_train, y_val, y_test, features = Titanic_Data_Preparation()

    # Initialize an empty dictionary
    hyperparameter = {}
    max_testing_accuracy=0

    #Now with the for Loop :
    for C in C_values:
        for tolerance in tol_values:
            for cache in cache_sizes:
                for iterations in max_iters:


                    model = SVC(C=C, kernel=kernel, degree=degree, gamma='scale', coef0=0.0,shrinking=True,
                                probability=False, tol=tolerance, cache_size=cache, class_weight=None, verbose=False,
                                max_iter=iterations,decision_function_shape='ovr', break_ties=False, random_state=None)
                    model.fit(X_train,y_train)
                    training_accuracy = evaluate_model(model, X_train, y_train)
                    cv_accuracy = evaluate_model(model, X_val, y_val)
                    testing_accuracy = evaluate_model(model, X_test, y_test)
                    metric_reading = check_model_state(training_accuracy, testing_accuracy, tolerance=0.05)
                    # print(f"Testing {kernel} for C {C} Tolerance {tolerance} cache_size {cache} max_itrations {iterations}")
                    # print(f" training accuracy  {training_accuracy}  testing accuracy  {testing_accuracy}")

                    # Store results in the dictionary
                    hyperparameter[(C, tolerance, cache, iterations)] = {
                        'training_accuracy': training_accuracy,
                        'testing_error': testing_accuracy,
                        'metric_reading': metric_reading,
                        'C_value': C,
                        'tolerance':tolerance,
                        'cache': cache,
                        'iterations':iterations
                    }
                    if testing_accuracy > max_testing_accuracy and abs(metric_reading)<0.05:
                        # print(f"Testing {kernel} for C {C} Tolerance {tolerance} cache_size {cache} max_itrations {iterations}")
                        # print(f" training accuracy  {training_accuracy}  testing accuracy  {testing_accuracy}")
                        check_model_state(training_accuracy,testing_accuracy , printing=True)
                        max_testing_accuracy = testing_accuracy
                        key = (C,tolerance,cache,iterations)
                        optimal_record = hyperparameter[key]


    #Who won the test :
    value = hyperparameter[key]
    if printing:
        print(f"The parameters that gave the best results  {key}")
        print(f"Having C of {key[0]} tolerance {key[1]} cache {key[2]} max_iterations {key[3]}")
        print(f"Having training accuracy {value['training_accuracy']}")
        print(f"Having testing accuracy {value['testing_error']}")
        print(f"Having metric reading {value['metric_reading']}")
    return optimal_record , max_testing_accuracy


def kernel_parameters():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    optimal_records = []
    testing_accuracy_list = []
    for kernel in kernels:
        optimal_record , testing_accuracy = find_best_hyperparameters(kernel)
        optimal_records.append(optimal_record)
        testing_accuracy_list.append(testing_accuracy)
    for i in range(len(optimal_records)):
        print(f"The kernel {kernels[i]} has optimal record of {optimal_records[i]}")
        print(f"The testing accuracy is {testing_accuracy_list[i]}")



def find_degree_polynomial():
    kernel = 'poly'
    degrees = [1,2,3,4,5,6]
    optimal_records = []
    testing_accuracy_list = []
    for degree in degrees:
        optimal_record, testing_accuracy = find_best_hyperparameters(kernel,degree)
        optimal_records.append(optimal_record)
        testing_accuracy_list.append(testing_accuracy)
    for i in range(len(optimal_records)):
        print(f"The kernel {kernel} has optimal record of {optimal_records[i]}")
        print(f"The testing accuracy is {testing_accuracy_list[i]}")



#Now we want to create a generic procedure for all kernels to decide their needs in terms of hyperparameters:
#The Procedure is as following.
#L2 Sweep . Keep all values default but change L2
#tolerance sweep
#cache_size sweep
#
#

def Profile_kernel():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Profiles = {}

    for i in range(len(kernels)):
        L2_optimal_record , L2_testing_accuracy = find_best_hyperparameters(kernels[i],skip_Tolerance=True,skip_max_iters=True,skip_cache_sizes=True)
        tolerance_optimal_record , tolerance_testing_accuracy = find_best_hyperparameters(kernels[i],skip_C=True,skip_max_iters=True,skip_cache_sizes=True)
        Cache_optimal_record , Cache_testing_accuracy = find_best_hyperparameters(kernels[i],skip_C=True,skip_max_iters=True,skip_Tolerance=True)
        iterations_optimal_record , iterations_testing_accuracy = find_best_hyperparameters(kernels[i],skip_C=True,skip_cache_sizes=True,skip_Tolerance=True)
        Profiles[(kernels[i])] = {
                        'L2_optimal_record': L2_optimal_record,
                        'L2_testing_accuracy': L2_testing_accuracy,
                        'tolerance_optimal_record': tolerance_optimal_record,
                        'tolerance_testing_accuracy':tolerance_testing_accuracy,
                        'Cache_optimal_record': Cache_optimal_record,
                        'Cache_testing_accuracy':Cache_testing_accuracy,
                        'iterations_optimal_record' : iterations_optimal_record,
                        'iterations_testing_accuracy':iterations_testing_accuracy
                    }
        #Now we should present the Profile of Each Kernel
        print(f"The kernel : {kernels[i]} Has the following Characteristics: ")
        print(f"    In the L2 Sweep : {L2_optimal_record['C_value']} testing accuracy : {L2_testing_accuracy}")
        print(f"    In the Tolerance Sweep: {tolerance_optimal_record['tolerance']} testing accuracy {tolerance_testing_accuracy}")
        print(f"    In the Cache Sweep: {Cache_optimal_record['cache']} testing accuracy {Cache_testing_accuracy}")
        print(f"    In the Iterations Sweep: {iterations_optimal_record['iterations']} testing accuracy {iterations_testing_accuracy}")

    return Profiles


Profile_kernel()