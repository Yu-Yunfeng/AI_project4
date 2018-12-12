from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score,precision_score,accuracy_score;

import dacision_tree as dt;
import pandas as pd
from sklearn.preprocessing import label_binarize;
import numpy as np
import sys

def avg_precision(y_test,y_pre,filename):
    precision = dict();
    classes = {"iris.data.txt":['Iris-setosa','Iris-versicolor','Iris-virginica'],
               "scale1.data.txt":[0,1,2]}
    Y_test = label_binarize(y_test,classes=classes[filename]);
    Y_pre = label_binarize(y_pre,classes=classes[filename]);
    for i in range(3):
        precision[i] = average_precision_score(Y_test[:,i],Y_pre[:,i]);
    return precision;

if __name__ == "__main__":
    n_splits = int(sys.argv[3]);
    test_size = float(sys.argv[2]);
    filename = sys.argv[1];
    classes = {"iris.data.txt":['Iris-setosa','Iris-versicolor','Iris-virginica'],
               "scale1.data.txt":[0,1,2]}
#    filename = "scale1.data.txt"
    data = pd.read_csv(filename,header=None);

    sss = StratifiedShuffleSplit(n_splits= n_splits, test_size=test_size);
    X = data.iloc[:,:-1];
    y = data.iloc[:,-1:];
    total_pre = 0.0;
    total_acc = 0.0;
    for train_indices , test_indices in sss.split(X,y):
        train_f = X.loc[train_indices];
        train_l = y.loc[train_indices];
        test_f = X.loc[test_indices];
        test_l = y.loc[test_indices];
        train_set = data.loc[train_indices];
        test_set = data.loc[test_indices];
        train_set.reset_index(inplace=True);
        train_set.drop(labels=['index'],inplace = True,axis = 1)

        decision_tree = dt.Decision_tree(filename);
        decision_tree.reload_data(train_set);
        decision_tree.run();
        pre_l = decision_tree.predict(test_set.values.tolist());
        pre_l_binarized = label_binarize(pre_l,classes=classes[filename]);
        test_l_binarized = label_binarize(test_l,classes=classes[filename]);

        acc_score = accuracy_score(test_l,pre_l);
#        print(acc_score)
        avg_pre_score = average_precision_score(test_l_binarized,pre_l_binarized);
        avg_presision = avg_precision(test_l,pre_l_binarized,filename)
        total_pre+=avg_pre_score
        print("the accuracy_score:",acc_score,"the avarage_precision_score_score:",avg_pre_score," the precision_score of each class :",avg_presision);
        total_acc+=acc_score;
    print("after",n_splits,"times, the average precision_score is",total_pre/n_splits," the average accuracy_score is",total_acc/n_splits);
