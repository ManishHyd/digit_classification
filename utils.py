from sklearn.model_selection import train_test_split
# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm

def findacc(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test,predicted)

def tune_hparams(model,X_train, X_test, X_dev , y_train, y_test, y_dev,list_of_param_combination):
    best_acc = -1
    for param_group in list_of_param_combination:
        temp_model = model(**param_group)
        temp_model.fit(X_train,y_train)
        acc = findacc(temp_model,X_dev,y_dev)
        if acc > best_acc:
            best_acc = acc
            best_model = temp_model
            optimal_param = param_group
    train_acc= findacc(best_model,X_train,y_train) 
    dev_acc = findacc(best_model,X_dev,y_dev)
    test_acc =  findacc(best_model,X_test,y_test)
    return train_acc, dev_acc, test_acc, optimal_param
    