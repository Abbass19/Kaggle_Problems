import numpy as np
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


from Problems.Problem_One_Titanic.utils.Data_loader import Titanic_Data_Preparation
from Problems.Problem_One_Titanic.utils.evaluation import evaluate_model


def Add_Noise(Input_Data, coefficient):
    Input_Data = np.array(Input_Data, dtype=np.float64)
    noise = np.random.normal(loc=0, scale=coefficient, size=Input_Data.shape)
    noisy_data = Input_Data + noise
    print(f"The data has data type of : {noisy_data.dtype}")
    return noisy_data




def train_decision_tree(X_train, X_test, y_train, y_test, max_depth=5):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    print("Decision Tree Results:")
    evaluate_model(model, X_test, y_test)

#DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
# class_weight=None, ccp_alpha=0.0, monotonic_cst=None)

def FindOptimalTree():
    #Getting the Input_Data:
    X_train, X_val, X_test, y_train, y_val, y_test, features = Titanic_Data_Preparation()

    #Setting the options to iterate over
    criterion = ['gini', 'entropy']
    max_depth = [3, 5,8,10,12, 15]
    min_samples_split = [2, 5,8,10]
    min_samples_leaf = [1, 2, 5]
    max_leaf_nodes = [None, 10, 20, 50]
    min_impurity_decrease = [0, 0.01, 0.05]

    #The variable to save parameters
    gini_hyperparameter = np.zeros((0,6))
    entropy_hyperparameter = np.zeros((0,6))


    #Here we will make 6 nested for loops
    #We will make them two loops because of the gini entropy thing
    criteria = 'gini'

    for depth in max_depth:
        for samples_split in min_samples_split:
            for sample_leaf in min_samples_leaf:
                for leaf_node in max_leaf_nodes:
                    for gain in min_impurity_decrease:

                        model = DecisionTreeClassifier(max_depth=depth, criterion=criteria, min_samples_split=samples_split,
                                                        min_samples_leaf=sample_leaf, max_leaf_nodes= leaf_node, min_impurity_decrease=gain)
                        model.fit(X_train, y_train)
                        train_accuracy = evaluate_model(model, X_train, y_train)
                        cv_accuracy = evaluate_model(model, X_val, y_val)
                        test_accuracy = evaluate_model(model, X_test, y_test)
                        new_record = np.array([depth, samples_split, sample_leaf, leaf_node,gain, test_accuracy])
                        gini_hyperparameter = np.vstack([gini_hyperparameter, new_record])
                        print(f"Training Model with criteria {criteria} depth {depth} sample split {samples_split} "
                                  f"sample leaf {sample_leaf} leaf node {leaf_node} minimum gain {gain} accuracy of {test_accuracy}")
                        check_model_state(train_accuracy,test_accuracy,tolerance=0.05)
    criteria = 'entropy'
    for depth in max_depth:
        for samples_split in min_samples_split:
            for sample_leaf in min_samples_leaf:
                for leaf_node in max_leaf_nodes:
                    for gain in min_impurity_decrease:

                        model = DecisionTreeClassifier(max_depth=depth, criterion=criteria, min_samples_split=samples_split,
                                                        min_samples_leaf=sample_leaf, max_leaf_nodes= leaf_node, min_impurity_decrease=gain)
                        model.fit(X_train, y_train)
                        train_accuracy = evaluate_model(model, X_train, y_train)
                        cv_accuracy = evaluate_model(model, X_val, y_val)
                        test_accuracy = evaluate_model(model, X_test, y_test)
                        new_record = np.array([depth, samples_split, sample_leaf, leaf_node,gain, test_accuracy])
                        entropy_hyperparameter = np.vstack([entropy_hyperparameter, new_record])
                        print(f"Training Model with criteria {criteria} depth {depth} sample split {samples_split} "
                                  f"sample leaf {sample_leaf} leaf node {leaf_node} minimum gain {gain} accuracy of {test_accuracy}")
                        check_model_state(train_accuracy, test_accuracy, tolerance=0.05)

    max_index_gini = np.argmax(gini_hyperparameter[:, -1])
    max_record_gini = gini_hyperparameter[max_index_gini]
    max_value_gini = max(gini_hyperparameter[:,-1])

    max_index_entropy = np.argmax(entropy_hyperparameter[:, -1])
    max_record_entropy = entropy_hyperparameter[max_index_entropy]
    max_value_entropy = max(entropy_hyperparameter[:,-1])

    if max_value_gini > max_value_entropy:
        optimal_record = max_record_gini
        samir = 'gini'
    else:
        optimal_record = max_record_entropy
        samir = 'entropy'


    mean_testing_error_gini = np.mean(gini_hyperparameter[:,-1])
    mean_testing_error_entropy = np.mean(entropy_hyperparameter[:,-1])
    mean_testing_error = (mean_testing_error_gini + mean_testing_error_entropy)/2

    gini_worst_case = min(gini_hyperparameter[:,-1])
    entropy_worst_case = min(entropy_hyperparameter[:,-1])
    worst_case = min(gini_worst_case,entropy_worst_case)

    print(f"The winner of the Decision Tree Problem is:")
    print(f"Training Model with criteria {samir} depth {optimal_record[0]} sample split {optimal_record[1]}"
          f"sample leaf {optimal_record[2]} leaf node {optimal_record[3]} minimum gain {optimal_record[4]}")
    print(f"\nHaving a testing error of {optimal_record[5]}")
    print(f"Overpassing the mean of models that is {mean_testing_error}")
    print(f"The Worst is {worst_case}")


def check_model_state(training_accuracy, testing_accuracy, baseline=None, tolerance=0.1):
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
    print(f"Status: {state}, Reading: {reading} Shifted by {percentage_shift*100}%")
    return percentage_shift


def DepthGraph(depth_limit, Noise=0):
    #This is also a procedure not a function
    # Getting the Input_Data:
    X_train, X_val, X_test, y_train, y_val, y_test, features = Titanic_Data_Preparation()

    if Noise!=0:
        X_train = Add_Noise(X_train,Noise)
        X_val = Add_Noise(X_val,Noise)
        X_test = Add_Noise(X_test,Noise)
        # y_train = Add_Noise(y_train,Noise) :) hahahahahahha WHAT THE HELL
        # y_val = Add_Noise(y_val,Noise)
        # y_test = Add_Noise(y_test,Noise)

    # Setting the options to iterate over
    max_depth = np.arange(1, depth_limit, 1)
    training_accuracy_list = []
    cv_accuracy_list = []
    testing_accuracy_list = []
    metric_Reading = []

    for depth in  max_depth:
        #Training Model for different Depth
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        training_accuracy = evaluate_model(model,X_train,y_train)
        cv_accuracy = evaluate_model(model,X_val,y_val)
        testing_accuracy = evaluate_model(model,X_test,y_test)
        metric_Reading.append(check_model_state(training_accuracy,testing_accuracy, tolerance=0.05))

        #Appending to the Lists
        training_accuracy_list.append(training_accuracy)
        cv_accuracy_list.append(cv_accuracy)
        testing_accuracy_list.append(testing_accuracy)

    # Find intersection point
    threshold = 0.1
    x_intersection = None
    for i in range(len(metric_Reading)):
        if metric_Reading[i] >= threshold:
            x_intersection = max_depth[i]
            break

    fig , axis = plt.subplots(1,2)
    axis[0].plot(max_depth,training_accuracy_list,label='training accuracy')
    axis[0].plot(max_depth,cv_accuracy_list , label = ' CV accuracy')
    axis[0].plot(max_depth,testing_accuracy_list, label = 'testing accuracy')
    axis[0].set_title(f"Error vs Tree Depth with Error of {Noise}")
    axis[1].set_title(f"Error vs Tree Depth with Error of {Noise}")
    axis[1].plot(max_depth,metric_Reading, label = 'Metric Reading')
    axis[1].axhline(0.1, color='red', linestyle='--', label='Threshold = 0.1')
    axis[1].axvline(x_intersection, color='blue', linestyle='--', label=f'Intersection at x = {x_intersection}')
    axis[1].text(2.5, 0.11, 'Over-fitting above 0.1\nGood below 0.1',
                 color='red', fontsize=10, horizontalalignment='center')
    axis[1].legend()
    axis[1].set_title("Metric Reading")
    axis[1].set_xlabel("Max Depth")
    axis[1].set_ylabel("Metric")
    plt.show()


def min_gain_Error():
    # This is also a procedure not a function
    # Getting the Input_Data:
    X_train, X_val, X_test, y_train, y_val, y_test, features = Titanic_Data_Preparation()

    # Setting the options to iterate over
    min_gain_list = np.linspace(0, 0.1, 100)
    training_accuracy_list = []
    cv_accuracy_list = []
    testing_accuracy_list = []
    metric_Reading = []
    for min_gain in  min_gain_list:
        #Training Model for different Depth
        model = DecisionTreeClassifier(min_impurity_decrease = min_gain, random_state=42)
        model.fit(X_train, y_train)
        training_accuracy = evaluate_model(model,X_train,y_train)
        cv_accuracy = evaluate_model(model,X_val,y_val)
        testing_accuracy = evaluate_model(model,X_test,y_test)
        metric_Reading.append(check_model_state(training_accuracy,testing_accuracy, tolerance=0.05))
        print(f"with min gain {min_gain}")

        #Appending to the Lists
        training_accuracy_list.append(training_accuracy)
        cv_accuracy_list.append(cv_accuracy)
        testing_accuracy_list.append(testing_accuracy)

# Find intersection point
    threshold = 0.05
    x_intersection = None
    for i in range(len(metric_Reading)):
        if metric_Reading[i] >= threshold:
            x_intersection = min_gain_list[i]
            break

    fig , axis = plt.subplots(1,2)
    axis[0].plot(min_gain_list,training_accuracy_list,label='training accuracy')
    axis[0].plot(min_gain_list,cv_accuracy_list , label = ' CV accuracy')
    axis[0].plot(min_gain_list,testing_accuracy_list, label = 'testing accuracy')
    axis[0].legend()
    axis[1].set_title("Error vs Tree Depth")
    axis[1].plot(min_gain_list,metric_Reading, label = 'Metric Reading')
    axis[1].axhline(0.1, color='red', linestyle='--', label='Threshold = 0.1')
    axis[1].axvline(x_intersection, color='blue', linestyle='--', label=f'Intersection at x = {x_intersection}')
    axis[1].text(2.5, 0.11, 'Over-fitting above 0.1\nGood below 0.1',
                 color='red', fontsize=10, horizontalalignment='center')
    axis[1].legend()
    axis[1].set_title("Metric Reading")
    axis[1].set_xlabel("Max Depth")
    axis[1].set_ylabel("Metric")
    plt.show()


def min_sample_heatmap():
    # This is also a procedure not a function
    # Getting the Input_Data:
    X_train, X_val, X_test, y_train, y_val, y_test, features = Titanic_Data_Preparation()

    # Setting the options to iterate over
    min_samples_split = np.arange(2, 16)
    min_samples_leaf =  np.arange(2, 16)
    Heat_Map_testing_accuracy = np.zeros((len(min_samples_split),len(min_samples_leaf)))
    Heat_Map_metric_reading = np.zeros((len(min_samples_split),len(min_samples_leaf)))


    #Data features[sample_split , sample_leaf ,train_accuracy, cv_accuracy, test_accuracy ,metric_reading ]
    #For the for loop we will have mapping
    # i --> sample split sample_split[i]
    # j --> sample_leaf  sample_leaf[j]
    for i in range(len(min_samples_split)):
        for j in range(len(min_samples_leaf)):
            # Training Model for different Depth
            model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf[j],min_samples_split=min_samples_split[i] ,random_state=42)
            model.fit(X_train, y_train)
            training_accuracy = evaluate_model(model, X_train, y_train)
            cv_accuracy = evaluate_model(model, X_val, y_val)
            testing_accuracy = evaluate_model(model, X_test, y_test)
            Heat_Map_testing_accuracy[i,j] = testing_accuracy
            Heat_Map_metric_reading[i,j] = check_model_state(training_accuracy, testing_accuracy, tolerance=0.05)
            print(f"With testing accuracy of {testing_accuracy}")
            #Decision Made:
            #We know this Input_Data is not sorted, we will try to generate the heatmap
            #If it worked we will postpone this problem.


    #Applying a threshold to Over-fitting Metric, We refuse models that are overfit. We only deal with good ones
    Mask_Zero = (Heat_Map_metric_reading > 0.05)
    Mask_One = (Heat_Map_metric_reading < 0.05)
    Heat_Map_metric_reading[Mask_Zero] = 0
    Heat_Map_metric_reading[Mask_One]  = 1
    Heat_Map_testing_accuracy_edited = Heat_Map_testing_accuracy *  Heat_Map_metric_reading

    fig,axis = plt.subplots(1,2)
    # Plot heatmap
    img1 = axis[0].imshow(Heat_Map_testing_accuracy, cmap='viridis', interpolation='nearest')
    fig.colorbar(img1, ax=axis[0], label='Accuracy')  # Add a color bar
    axis[0].set_title("Heat map for Data without Filtering")
    axis[0].set_xlabel("Minimum Samples Split")
    axis[0].set_ylabel("Minimum Samples Leaf")

    # Plot heatmap
    img2 = axis[1].imshow(Heat_Map_testing_accuracy_edited, cmap='viridis', interpolation='nearest')
    fig.colorbar(img2, ax=axis[1], label='Accuracy')  # Add a color bar
    axis[1].set_title("Heat map for Data with Filtering")
    axis[1].set_xlabel("Minimum Samples Split")
    axis[1].set_ylabel("Minimum Samples Leaf")

    plt.tight_layout()
    plt.show()




def Noise_Effect():
    #This is a procedure as well
    #We will try to imitate the effect of noisy data on model's learning
    #For that we will use the Depth function after we added tje noise thing
    noise = np.linspace(0,3,30)
    for i in range(len(noise)):
        print(f"This graph is with Error SNR of {noise[i]}")
        DepthGraph(41, noise[i])



def Optimizer():
    #This optimizer will create multiple decision tree and return the most accurate one that is Not Over-fitting
    models = []
    testing_accuracy_list = []
    Reading = []

    # Initialization = [max_depth,max_leaf_nodes ,min_impurity_decrease]
    #Hyperparameters
    #Accelerators (maximum leaf nodes - decrease ;min_impurity_decrease - decrease ;  max_depth - increase , )

    #Regulators [max_depth - decrease , min_impurity_decrease - increase]
    number_steps = 1000
    max_depth = np.linspace(3,15,number_steps)
    max_leaf_nodes = np.arange(0,number_steps)
    min_impurity_decrease = np.linspace(0,0.2,number_steps)

    # Calculate step sizes
    step_max_depth = max_depth[1] - max_depth[0]
    step_max_leaf_nodes = max_leaf_nodes[1] - max_leaf_nodes[0]
    step_min_impurity_decrease = min_impurity_decrease[1] - min_impurity_decrease[0]

    #Initializations and HyperParameter

    Variable_max_depth = 5
    Variable_max_leaf_nodes = 2
    Variable_min_impurity = .3

    Int_max_depth = np.round(Variable_max_depth)
    Int_max_leaf_nodes = np.round(Variable_max_leaf_nodes)


    #Array Ratio Issue :
    Over_fitting_ratio  = np.array([-2/3, -1/3, 0 ])
    Under_fitting_ratio = np.array([0 ,1/3 , -2/3])
    reading = 0.1

    #Start training the Model
    X_train, X_val, X_test, y_train, y_val, y_test, features = Titanic_Data_Preparation()
    model = DecisionTreeClassifier(max_depth=Int_max_depth, min_impurity_decrease=Variable_min_impurity, max_leaf_nodes=Int_max_leaf_nodes)
    model.fit(X_train,y_train)
    training_accuracy = evaluate_model(model,X_train,y_train)
    cv_accuracy = evaluate_model(model,X_val,y_val)
    testing_accuracy = evaluate_model(model,X_test,y_test)
    testing_accuracy_list.append(testing_accuracy)
    models.append(model)
    reading = check_model_state(training_accuracy,testing_accuracy,tolerance=0.05)
    Reading.append(reading)



    while abs(reading)>0.01 and testing_accuracy<0.8:
        steps = np.round(abs(reading) * 40)
        steps = int(steps)


        if reading>0:
            #We have Over-fitting Problem
            #Update Variables
            Variable_max_depth += (-2/3)*steps*step_max_depth
            Variable_max_leaf_nodes += (-1/3)*steps*step_max_leaf_nodes

            Int_max_depth = int(np.round(Variable_max_depth))
            Int_max_leaf_nodes = int(np.round(Variable_max_leaf_nodes))




        else:
            #We Have Under-fitting
            #Update the variables
            Variable_max_leaf_nodes += (+1 / 3) * steps * step_max_leaf_nodes
            Variable_min_impurity += (-2 / 3) * steps * step_min_impurity_decrease

            Int_max_leaf_nodes = int(np.round(Variable_max_leaf_nodes))



        # Start training the Model In the Loop :
        model = DecisionTreeClassifier(max_depth=Int_max_depth, min_impurity_decrease=Variable_min_impurity,
                                           max_leaf_nodes=Int_max_leaf_nodes)
        model.fit(X_train, y_train)
        training_accuracy = evaluate_model(model, X_train, y_train)
        cv_accuracy = evaluate_model(model, X_val, y_val)
        testing_accuracy = evaluate_model(model, X_test, y_test)
        #Saving the Model
        testing_accuracy_list.append(testing_accuracy)
        models.append(model)

        reading = check_model_state(training_accuracy, testing_accuracy, tolerance=0.05)
        print(f"Trained Parameters : depth : {Int_max_depth} min_gain{Variable_min_impurity} max_leaf_nodes : {Int_max_leaf_nodes} ")
        print(f"Has reading of {reading} ")
        print(f"The model has testing_accuracy {testing_accuracy}")
        Reading.append(reading)















#MAIN :
#Work ALl the functions and Save the Images:
Optimizer()

# FindOptimalTree()
# DepthGraph(30)
# min_gain_Error()
# min_sample_heatmap()
# Noise_Effect()
