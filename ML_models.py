from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from os import listdir
#from os.path import isfile, join
import cv2
import numpy as np

from sklearn.model_selection import LeaveOneOut

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout

from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

### SVM Imports ###
from sklearn.svm import SVC
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

#K-fold and LOO Cross Validation Imports
from sklearn.model_selection import LeaveOneOut,KFold

#CNN Validation Imports
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
import tensorflow

#X_sample = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[1,2,3],[4,5,6],[7,8,9],[10,11,12],[1,2,3],[4,5,6],[7,8,9],[10,11,12],[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#y_sample = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
#print(X_sample.shape)
#print(y_sample.shape)

### FUNCTIONS TO PROCESS MODELS ###
def apply_holdout(data,target,training_ratio=0.7):
    '''
    Applies holdout on dataset: assumes a 70/30 split
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1-training_ratio, stratify=target, random_state=0)
    return [X_train,X_test,y_train,y_test]

#output_tensor = apply_holdout(X_sample,y_sample,0.5)
#print("-------------------------")
#print("X_train:",output_tensor[0])
#print("X_test:",output_tensor[1])
#print("y_train:",output_tensor[2])
#print("y_test",output_tensor[3])

def apply_LOO_CV(data, labels):
  '''
  Function to apply leave one out cross validation on input data
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

#output_tensor = apply_LOO_CV(X_sample,y_sample)
#print("-------------------------")
#print(type(output_tensor))
#print(output_tensor)
#print("X_train shape:",len(output_tensor[0]))
#print("X_train 0 shape:",output_tensor[0][0].shape)
#print("X_test:",len(output_tensor[1]))
#print("X_test 0 shape:",output_tensor[1][0].shape)
#print("y_train:",len(output_tensor[2]))
#print("y_train 0 shape:",output_tensor[2][0].shape)
#print("y_test",len(output_tensor[3]))
#print("y_test 0 shape:",output_tensor[3][0].shape)

# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# Reference: https://machinelearningmastery.com/k-fold-cross-validation/

def apply_KF_CV(data, labels):
  '''
  Function to apply k-fold cross validation on input data
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

def categorize_labels(y_train,y_test,num_classes):
    '''
    One Hot encodes row vectors
    '''
    y_train_cat = to_categorical(y_train,num_classes)
    y_test_cat = to_categorical(y_test,num_classes)

    return y_train_cat,y_test_cat

################## MLP CLASS ################

class MLP_Class():
    '''
    This class is used when working with the Multilayer Perceptron Model
    '''
    #Define variables
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.num_features = data.shape[1] #Store the number of input features in the data
        self.num_classes = len(np.unique(target)) #Store the number of classes in the target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_cat = None
        self.y_test_cat = None
        self.X_train_list = []
        self.X_test_list = []
        self.y_train_list = []
        self.y_test_list = []
        self.num_folds = None
        self.y_train_cat_list = []
        self.y_test_cat_list = []
        self.folds = []

    #Function for holodout split
    def initialize_MLP(self):
        self.X_train,self.X_test,self.y_train,self.y_test = apply_holdout(self.data,self.target)
        self.y_train_cat,self.y_test_cat = categorize_labels(self.y_train,self.y_test,self.num_classes)

    #Function for leave one out split
    def initialize_MLP_LOO(self):

        self.X_train_list,self.X_test_list,self.y_train_list,self.y_test_list = apply_LOO_CV(self.data,self.target)
        print(len(self.X_train_list),",",len(self.X_test_list),",",len(self.y_train_list),",",len(self.y_test_list))
        self.num_folds = len(self.X_train_list)

        #This loop combines all of the folds required for leave one out
        self.folds = []
        for i in range(self.num_folds):
            y_train_cat,y_test_cat = categorize_labels(self.y_train_list[i],self.y_test_list[i],self.num_classes)
            self.y_train_cat_list.append(y_train_cat)
            self.y_test_cat_list.append(y_test_cat)
            new_fold = [self.X_train_list[i],self.X_test_list[i],self.y_train_list[i],self.y_test_list[i],y_train_cat,y_test_cat]
            self.folds.append(new_fold)

    def construct_MLP_model(self,hyperparams):
        '''
        hyperparams: tuple of hyperparameters used to configure different MLP models
        '''
        hyp = hyperparams

        model=Sequential() #Creates a sequential model

        model.add(Dense(hyp[0], activation=hyp[1], input_dim=self.num_features)) #First layer (INPUT LAYER): 256 nodes, input is the 2 PCA-reduced feature vectors, activation
                                                               # function is 'relu'. Note: input only needs to be specified for first layer.
        model.add(Dropout(hyp[2])) #With each iteration, 20% of the neurons of the input layer will 'dropout'.

        model.add(Dense(hyp[3], activation=hyp[4])) #Hiddlen Layer 1 with 128 neurons.
        model.add(Dropout(hyp[5])) #10% of this hidden layer is likely to drop out on each iteration.

        model.add(Dense(hyp[6], activation=hyp[7])) #Hidden Layer 2 with 128 neurons.
        model.add(Dropout(hyp[8])) #5% of this hidden layer is likely to drop out on each iteration.

        model.add(Dense(2, activation='softmax')) #Output layer of size 2 (1 for each class).

        epochs=100 #NOTE: 1 epoch is 1 entire pass of the training data through the algorithm (ie. all 280 samples need to go through to complete 1 epoch)
        batch_size=128 #Input has 280 samples: so requires 3 batches. Each batch does forward/backward pass and updates weights. Ie. Loss function computation/
                        # gradient descent is applied only after each batch.
        red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=2, verbose=1)
        model.compile(optimizer=Adam(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

        return model

    #This function iterates through the hyperparameter set 
    def run_model_n_times_per_hyperparam_set(self,hyperparams,n=1):

        train_loss_vec = []
        train_acc_vec = []
        val_loss_vec = []
        val_acc_vec = []

        for i in range(n):
            #print("------------ Iteration " + str(i) + " ------------")

            model = self.construct_MLP_model(hyperparams)
            #model.summary()
            #print("X.train:",self.X_train)
            #print("y_train_cat:",self.y_train_cat)
            #print("X.test:",self.X_test)
            #print("y_test_cat:",self.y_test_cat)

            history_temp = model.fit(self.X_train,self.y_train_cat, epochs = 100, validation_data = (self.X_test,self.y_test_cat),batch_size=128, verbose = 0)

            train_loss_vec.append(history_temp.history['loss'][-1])
            train_acc_vec.append(history_temp.history['accuracy'][-1])
            val_loss_vec.append(history_temp.history['val_loss'][-1])
            val_acc_vec.append(history_temp.history['val_accuracy'][-1])

        #print("Training Loss Vector: ",train_loss_vec)
        #print("Training Accuracy Vector: ",train_acc_vec)
        #print("Validation Loss Vector: ",val_loss_vec)
        #print("Validation Accuracy Vector: ",val_acc_vec)

        train_loss_ave = sum(train_loss_vec)/n
        train_acc_ave = sum(train_acc_vec)/n
        val_loss_ave = sum(val_loss_vec)/n
        val_acc_ave = sum(val_acc_vec)/n

        #print("Model Statistics: ")
        #print("Training Loss Average: ",train_loss_ave)
        #print("Training Accuracy Average: ",train_acc_ave)
        #print("Validation Loss Average: ",val_loss_ave)
        #print("Validation Accuracy Average: ",val_acc_ave)

        return train_loss_ave,train_acc_ave,val_loss_ave,val_acc_ave

    def variate_hyperparams_on_MLP_model(self,combination_list,n=1):
        '''
        combination_list: a list of all possible hyperparameter values for each hyperparameter; (list of tuples)
        '''

        #Open File to Write: Overwrite Existing Content
        f = open("MLP_Model_Runs.txt", "w")

        print("========== Step 1: Computing Combinations for Hyperparams ==========")

        combinations = [] #A list of possible hyperparam combinations.

        #Start with full list and remove an element at start of each loop
        while combination_list != []:
            #print("Outer iteration: ")
            tuple_elem = combination_list[0]
            #print("Tuple elem: ",tuple_elem)
            new_combinations = []

            #Each tuple contains possible choices for hyperparam
            for possible_choice in tuple_elem:
                if combinations != []:
                    #Add the possible choice to the end of each list in the combination_list
                    for hyperparam_tuple in combinations:
                        new_hyperparam_tuple = hyperparam_tuple + (possible_choice,)
                        new_combinations.append(new_hyperparam_tuple)
                else:
                    new_hyperparam_tuple = (possible_choice,)
                    new_combinations.append(new_hyperparam_tuple)

            combinations = new_combinations[:]
            combination_list.pop(0)

        print("Number of Hyperparam Combinations: ", len(combinations))
        f.write("Number of Hyperparam Combinations: "+ str(len(combinations)))
        print("\n")
        f.write("\n")
        print("Number of Runs per Hyperparam Set: ", str(n))
        f.write("Number of Runs per Hyperparam Set: "+str(n))
        print("\n")
        f.write("\n")
        print("========== Step 2: Running model n times for each hyperparam ==========")

        model_eval_dict = {}

        i = 1
        for hyperparam_tuple in combinations:
            print("Hyperparam Set " + str(i) + ":", hyperparam_tuple)
            train_loss_ave,train_acc_ave,val_loss_ave,val_acc_ave = self.run_model_n_times_per_hyperparam_set(hyperparam_tuple,n)
            model_eval_dict[hyperparam_tuple] = [train_loss_ave,train_acc_ave,val_loss_ave,val_acc_ave]
            i += 1

        print("\nHyperparam Tuple | Training Loss Average | Training Accuracy Average | Validation Loss Average | Validation Accuracy Average")
        f.write("\n")
        f.write('{:60s} {:25s} {:25s} {:25s} {:20s}'.format("Hyperparam Tuple ","| Training Loss Avg ","| Training Accuracy Avg ","| Validation Loss Avg ","| Validation Acc Avg"))
        for elem in model_eval_dict.keys():
            print(str(elem) + " | " + str(model_eval_dict[elem][0]) + " | " + str(model_eval_dict[elem][1]) + " | " + str(model_eval_dict[elem][2]) + " | " + str(model_eval_dict[elem][3]))
            f.write("\n")
            term1 = str(elem)
            term2 = str("| " + str(model_eval_dict[elem][0]) + " ")
            term3 = str("| " + str(model_eval_dict[elem][1]) + " ")
            term4 = str("| " + str(model_eval_dict[elem][2]) + " ")
            term5 = str("| " + str(model_eval_dict[elem][3]))
            f.write('{:60s} {:25s} {:25s} {:25s} {:20s}'.format(term1,term2,term3,term4,term5))

        f.close()
        return model_eval_dict

    #This function seperates the train test split used for the MLP model
    def generate_MLP_models(self):

        print("Prior to Train Test Split")
        self.X_train,self.X_test,self.y_train,self.y_test = apply_holdout(0.7)
        num_classes = len(np.unique(self.y_test))
        self.y_train_cat,self.y_test_cat = categorize_labels(self.y_train,self.y_test,num_classes)
        hyperparam_values = [(128,256),('relu',),(0.1,0.2,0.5),(128,256),('relu',),(0.05,0.1,0.2),(128,256),('relu',),(0.05,0.1,0.2)] #216 PERMUTATIONS
        #hyperparam_values = [(256,),('relu',),(0.2,),(128,256),('relu',),(0.1,),(128,),('relu',),(0.05,)] #2 PERMUTATIONS
        #hyperparam_values = [(128,256),('relu',),(0.1,),(128,256),('relu',),(0.05,),(128,),('relu',),(0.05,0.1,0.2)]
        evaluated_models_dict = self.variate_hyperparams_on_MLP_model(hyperparam_values,5)
        self.evaluated_models_dict = evaluated_models_dict

    #This function prints the accuracy and losses of a test, train split
    def evaluate_model(self, hyperparam_eval_set, num_epochs = 100):

        model = self.construct_MLP_model(hyperparam_eval_set)

        History = model.fit(self.X_train,self.y_train_cat, epochs = num_epochs, validation_data = (self.X_test,self.y_test_cat),batch_size=128, verbose = 1) #Original Vectors

        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'test'])
        plt.show()

        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'test'])
        plt.show()

    #This function prints the accuracy and losses of using leave one out cross validation
    def evaluate_model_LOO(self, hyperparam_eval_set):

        train_loss_vec = np.array([])
        train_acc_vec = np.array([])
        val_loss_vec = np.array([])
        val_acc_vec = np.array([])

        for i in range(self.num_folds):
        #for i in range(20):
            print("Working on fold: ",i)
            model = self.construct_MLP_model(hyperparam_eval_set)

            X_train = self.X_train_list[i]
            y_train_cat = self.y_train_cat_list[i]
            X_test = self.X_test_list[i]
            y_test_cat = self.y_test_cat_list[i]
            History = model.fit(X_train,y_train_cat, epochs = 100, validation_data = (X_test,y_test_cat),batch_size=128, verbose = 0) #Original Vectors

            plt.plot(History.history['accuracy'])
            plt.plot(History.history['val_accuracy'])
            train_loss_vec = np.append(train_loss_vec,History.history['loss'][-1])
            train_acc_vec = np.append(train_acc_vec,History.history['accuracy'][-1])
            val_loss_vec = np.append(val_loss_vec,History.history['val_loss'][-1])
            val_acc_vec = np.append(val_acc_vec,History.history['val_accuracy'][-1])
            #print(train_loss_vec)
            #assert(False)
            #plt.title('Model Accuracy')
            #plt.ylabel('Accuracy')
            #plt.xlabel('Epochs')
            #plt.legend(['train', 'test'])
            #plt.show()

            #plt.plot(History.history['loss'])
            #plt.plot(History.history['val_loss'])
            #plt.title('Model Loss')
            #plt.ylabel('Loss')
            #plt.xlabel('Epochs')
            #plt.legend(['train', 'test'])
            #plt.show()
        train_loss_mean = np.mean(train_loss_vec)
        train_acc_mean = np.mean(train_acc_vec)
        val_loss_mean = np.mean(val_loss_vec)
        val_acc_mean = np.mean(val_acc_vec)

        train_loss_stddev = np.std(train_loss_vec)
        train_acc_stddev = np.std(train_acc_vec)
        val_loss_stddev = np.std(val_loss_vec)
        val_acc_stddev = np.std(val_acc_vec)

        print("Training Loss Mean: ",train_loss_mean)
        print("Training Acc Mean: ",train_acc_mean)
        print("Validation Loss Mean: ",val_loss_mean)
        print("Validation Acc Mean: ",val_acc_mean)

        print("Training Loss Stddev: ",train_loss_stddev)
        print("Training Acc Stddev: ",train_acc_stddev)
        print("Validation Loss Stddev: ",val_loss_stddev)
        print("Validation Acc Stddev: ",val_acc_stddev)

class SVC_Class():
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.SVC_kernels = {0:'linear',1:'poly',2:'rbf',3:'sigmoid'}

    def initialize_SVC(self):
        self.X_train,self.X_test,self.y_train,self.y_test = apply_holdout(self.data,self.target)

    def run_SVC(self,kernel_index):
        kernel_type = None
        kernel_type = self.SVC_kernels[kernel_index]

        print("------------- CLASSIFIER TYPE: ",kernel_type," ------------")
        classifier = SVC(kernel = kernel_type, random_state = 0)
        classifier.fit(self.X_train, self.y_train)

        #Predict Test Set Results
        y_pred = classifier.predict(self.X_test)

        #print("Predicted y: ",y_pred)
        #print("Actual y: ", y_test)

        #Make a confusion matrix/accuracy score
        cm = confusion_matrix(self.y_test, y_pred)
        #print(cm.shape)
        #print(cm)
        acc = accuracy_score(self.y_test,y_pred)
        #print(acc)

        #Heatmap
        #plt.figure(1, figsize=(16, 9))
        #sns.heatmap(confusion_matrix(y_test, y_pred))
        #plt.show()
        #print(classification_report(y_test, y_pred))
        #print("accuracy score:{:.2f}%".format(metrics.accuracy_score(y_test, y_pred)*100))
        return cm,acc

    def run_all_4_SVCs(self):
        '''
        Runs all 4 SVCs and returns the classification matrix of the best
        '''
        ret_list = []

        #Attempt all 4 SVCs
        for i in range(4):
            cm,acc = self.run_SVC(i)
            kernel_type = self.SVC_kernels[i]
            ret_list.append((kernel_type,cm,acc))

        return ret_list

class CNN_Class():

    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.num_features = data.shape[1] #Store the number of input features in the data
        self.num_classes = len(np.unique(target)) #Store the number of classes in the target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_cat = None
        self.y_test_cat = None
        self.X_train_list = []
        self.X_test_list = []
        self.y_train_list = []
        self.y_test_list = []
        self.num_folds = None
        self.y_train_cat_list = []
        self.y_test_cat_list = []
        self.folds = []

    def initialize_CNN(self):
        self.X_train,self.X_test,self.y_train,self.y_test = apply_holdout(self.data,self.target)
        self.y_train_cat,self.y_test_cat = categorize_labels(self.y_train,self.y_test,self.num_classes)

    def construct_and_run_CNN_model(self):
        '''
        hyperparams: tuple of hyperparameters used to configure different MLP models
        '''
        #References:
        # https://pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
        # https://medium.com/@msgold/predicting-images-with-a-cnn-90a25a9e4509#:~:text=In%20summary%2C%20a%20Convolutional%20Neural,in%20various%20computer%20vision%20tasks.

        conv_model = Sequential()
        conv_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1300,1254,1)))
        conv_model.add(MaxPooling2D((2, 2)))
        conv_model.add(Conv2D(64, (3, 3), activation='relu'))
        conv_model.add(MaxPooling2D((2, 2)))
        conv_model.add(Conv2D(64, (3, 3), activation='relu'))
        conv_model.add(Flatten())
        conv_model.add(Dense(128, activation='relu'))
        conv_model.add(Dense(2))

        conv_model.summary()

        conv_model.compile(optimizer='adam', loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

        print("Before Fit")

        History = conv_model.fit(self.X_train,self.y_train, epochs = 100,validation_data = (self.X_test,self.y_test),batch_size=128, verbose = 1)

        #y_pred_encoded = conv_model.predict(X_test)
        #print(y_pred_encoded.shape)
        #y_pred = np.argmax(y_pred_encoded, axis=1)
        #print(y_pred)
        #print(y_test)
