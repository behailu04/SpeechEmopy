import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.model_selection import KFold
import numpy as np
import keras
import csv
import pandas as pd
from keras.layers.core import Activation
def load_features():
    path = "rnn_features_pos_neu/"
    X_train = np.load(path + "train/train_x.npy")
    Y_train = np.load(path + "train/train_y.npy")
    X_test = np.load(path + "test/test_x.npy")
    Y_test = np.load(path + "test/test_y.npy")
    print X_train.shape
    print X_test.shape
    print Y_train.shape
    print Y_test.shape
    return X_train, Y_train, X_test, Y_test
def b_model():
    X, Y, X_test, Y_test = load_features()
    model = Sequential()
    model.add(LSTM(512,return_sequences=True, input_shape=(20, 41)))  
    model.add(Activation('tanh'))
    model.add(LSTM(256, return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dense(120))
    model.add(Activation('tanh'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # his = History((X_test, Y_test))
    history = model.fit(X, Y, batch_size = 120, epochs = 60, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    model_json = model.to_json()
    with open("model/rnn_pos_neu.json", "w") as json_file:
         json_file.write(model_json)
    model.save_weights("model/rnn_pos_neu.h5")
    print("Model saved to disk\n")
    print("Model has finished. Accuracy:" + str(score[1]) + " and loss:" + str(score[0]))
    file = open("rnn_pos_neu_metrics.txt", "w")
    los = "loss:"+str(score[0])
    acc = "accuracy:" + str(score[1])
    file.write(los)
    file.write(acc)
    file.close()
    return score
def build_model():
    X_train, Y_train, X_test, Y_test = load_features()
    accuracy = []
    loss = []
    samples = []
    tr_acc = []
    tr_loss = []
    for i in range(len(X_train)/500):
        print ("Modeling: " + str(i) + " of " + str(len(X_train)/500))
        x_t = X_train[:(i +1) * 500]
        y_t = Y_train[:(i + 1) *500]
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(20, 41)))  

        model.add(Dropout(0.2))

        # return a single vector of dimension 128
        model.add(LSTM(128))  
        model.add(Dropout(0.2))

        # apply softmax to output
        model.add(Dense(6, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        # his = History((X_test, Y_test))
        history = model.fit(x_t, y_t, batch_size = 100, epochs = 10, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("Model has finished. Accuracy:" + str(score[1]) + " and loss:" + str(score[0]))
        accuracy.append(score[1])
        loss.append(score[0])
        samples.append(x_t.shape[0])
        tr_acc.append(history.history['acc'][-1])
        tr_loss.append(history.history['loss'][-1])
    # f = open("rnn_test_accuracy.txt", "w+")
    # for x in accuracy:
    #     f.write(str(x) + ",")
    # f.close()
    # f = open("rnn_train_accuracy.txt", "w+")
    # for x in tr_acc:
    #     f.write(str(x) + ",")
    # f.close()

    # f = open("rnn_train_loss.txt", "w+")
    # for x in tr_loss:
    #     f.write(str(x) + ",")
    # f.close()

    # f = open("rnn_test_loss.txt", "w+")
    # for x in loss:
    #     f.write(str(x) + ",")
    # f.close()
    # print(accuracy)
    # print(tr_acc)
    # print(samples)
    # plt.plot(accuracy)
    # plt.plot(tr_acc)
    # plt.ylabel('accuracy')
    # plt.xlabel('Training sample')
    # plt.legend(['test', 'train'], loc='upper left')
    # plt.show()
    # print(loss)
    # print(tr_loss)
    # plt.plot(loss)
    # plt.plot(tr_loss)
    # plt.ylabel('Loss')
    # plt.xlabel('Training sample')
    # plt.legend(['test', 'train'], loc='upper left')
    # plt.show()
    # with open('rnn_accuracy.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(accuracy)
    # with open('rnn_loss.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(loss)
    # with open('rnn_samples.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(samples)
        
    # model_json = model.to_json()
    # with open("model/rnn5_model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("model/rnn5_model.h5")
    # print("Model saved to disk\n")
   
    # n_samples = X_train.shape[0]
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    # plt.plot(X_train.shape[0])
    # plt.plot(history.history['acc'])
    # plt.show()
    # plt.plot(samples, accuracy)
    # plt.plot(samples, loss)
    # plt.title('model accuracy and loss')
    # plt.ylabel('accuracy')
    # plt.xlabel('Training sample')
    # plt.legend(['accuracy', 'loss'], loc='upper left')
    # plt.show()
    
    
class History(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.accuracy = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []
    def on_batch_end(self, batch, logs = {}):
        self.accuracy.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        X, Y = self.test_data
        val_loss, val_acc = self.model.evaluate(X, Y, verbose=0)
        self.val_acc.append(val_acc)
        self.val_loss.append(val_loss)
    def get_accuracy_loss(self):
        return self.accuracy
    def get_val_acc_loss(self):
        return self.val_acc
    
# build_model()
b_model()


