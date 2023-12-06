from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from os import listdir
#from os.path import isfile, join
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout

from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class MLP():

    def __init__(self,data,target):
        self.X = data
        self.target = target
        self.num_features = data.shape[1]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_cat = None
        self.y_test_cat = None

    def perform_train_test_split(self,testing_ratio=0.7):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.target, test_size=testing_ratio, stratify=self.target, random_state=0)

    def categorize_target(self):
        #print("Y train:",y_train)
        self.y_train_cat=to_categorical(self.y_train,2) #Converts decimal classes into matrix of row vectors. Each row vector is a one-hot representation of the class.
        #print("Y train Cat:",y_train_cat)
        self.y_test_cat=to_categorical(self.y_test,2)

    def create_MLP_model(self,hyperparams):
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

    def run_model_n_times_per_hyperparam_set(self,hyperparams,n=1):

        train_loss_vec = []
        train_acc_vec = []
        val_loss_vec = []
        val_acc_vec = []

        for i in range(n):
            #print("------------ Iteration " + str(i) + " ------------")

            model = self.create_MLP_model(hyperparams)
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

    def generate_MLP_models(self):

        print("Prior to Train Test Split")
        self.perform_train_test_split(0.7)
        self.categorize_target()
        hyperparam_values = [(128,256),('relu',),(0.1,0.2,0.5),(128,256),('relu',),(0.05,0.1,0.2),(128,256),('relu',),(0.05,0.1,0.2)] #216 PERMUTATIONS
        #hyperparam_values = [(256,),('relu',),(0.2,),(128,256),('relu',),(0.1,),(128,),('relu',),(0.05,)] #2 PERMUTATIONS
        #hyperparam_values = [(128,256),('relu',),(0.1,),(128,256),('relu',),(0.05,),(128,),('relu',),(0.05,0.1,0.2)]
        evaluated_models_dict = self.variate_hyperparams_on_MLP_model(hyperparam_values,5)
        self.evaluated_models_dict = evaluated_models_dict

    def evaluate_model(self):
        hyperparam_eval_set = (256, 'relu', 0.1, 256, 'relu', 0.05, 128, 'relu', 0.2)
        model = self.create_MLP_model(hyperparam_eval_set)
        History = model.fit(self.X_train,self.y_train_cat, epochs = 100, validation_data = (self.X_test,self.y_test_cat),batch_size=128, verbose = 1) #Original Vectors

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



