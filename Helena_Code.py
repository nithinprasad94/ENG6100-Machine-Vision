from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.model_selection import KFold

def LOO_cross_validation(data, labels):
  '''
  Data is expected as a list of serialized images
  '''
  leave_one_out = LeaveOneOut()

  # Initialize lists to store data
  X_train_list = []
  X_test_list = []
  y_train_list = []
  y_test_list = []

  for train_index, test_index in leave_one_out.split(data):
    X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)

  return X_train_list, X_test_list, y_train_list, y_test_list

# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# Reference: https://machinelearningmastery.com/k-fold-cross-validation/

def KFOLD_cross_validation(data, labels):
  '''
  Data is expected as a list of serialized images
  '''
  kfold = KFold(n_splits=5, shuffle=False, random_state=None)

  X_train_list = []
  X_test_list = []
  y_train_list = []
  y_test_list = []

  for train_index, test_index in kfold.split(data):
    X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)

  return X_train_list, X_test_list, y_train_list, y_test_list


