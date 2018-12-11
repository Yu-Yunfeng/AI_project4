from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score,precision_score;

import dacision_tree as dt;
import pandas as pd
from sklearn.preprocessing import label_binarize;
##
def avg_precision(y_test,y_pre):
    precision = dict();
    Y_test = label_binarize(y_test,classes=['Iris-setosa','Iris-versicolor','Iris-virginica']);
    Y_pre = label_binarize(y_pre,classes=['Iris-setosa','Iris-versicolor','Iris-virginica']);
    for i in range(3):
        precision[i] = average_precision_score(Y_test[:,i],Y_pre[:,i]);
    return precision;

if __name__ == "__main__":
    filename = "iris.data.txt"
    data = pd.read_csv(filename,header=None);
    sss = StratifiedShuffleSplit(test_size=0.2);
    X = data.iloc[:,:-1];
    y = data.iloc[:,-1:];
#    print(data.iloc[:,-1:].nunique())

    for train_indices , test_indices in sss.split(X,y):
        train_f = X.loc[train_indices];
        train_l = y.loc[train_indices];
        test_f = X.loc[test_indices];
        test_l = y.loc[test_indices];
        train_set = data.loc[train_indices];
        test_set = data.loc[test_indices];


        train_set.reset_index(inplace=True);
        train_set.drop(labels=['index'],inplace = True,axis = 1)
#        print(train_set)
#        print(train_set)
        decision_tree = dt.Decision_tree(filename);
        decision_tree.reload_data(train_set);
        decision_tree.run();
        pre_l = decision_tree.predict(test_set.values.tolist());
        pre_l_binarized = label_binarize(pre_l,classes=['Iris-setosa','Iris-versicolor','Iris-virginica']);
        test_l_binarized = label_binarize(test_l,classes=['Iris-setosa','Iris-versicolor','Iris-virginica']);

        avg_pre_score = average_precision_score(test_l_binarized,pre_l_binarized);
#        pre_score = precision_score(test_l_binarized,pre_l_binarized)
        avg_presision = avg_precision(test_l,pre_l_binarized)
        print(avg_pre_score);
#        print(pre_score)
        print(avg_presision)