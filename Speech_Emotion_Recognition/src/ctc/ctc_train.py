import time
import numpy as np
import tensorflow as tf
import os
import sys
import csv
import time
import keras
import keras.backend as K
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import  Activation, Dropout
from keras.layers import LSTM, Input, Lambda, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Merge
from keras.layers import LSTM, Input, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
# Hapyer parametes
batch_size = 100
nb_feat = 13
nb_class = 6
nb_epoch = 30
optimizer = 'Adadelta'
emotions = ["Happy", "Sad", "Surprise", "Fear", "Angry", "Neutral"]
maxlen = 319
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    shift = 2
    y_pred = y_pred[:, shift:, :]
    input_length -= shift
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_sample():
    print("loading features...")
    names = os.listdir("../dataset/features/")
    tx = []
    ty = []
    for file in names:
        x, y = load_sample(file)
        tx.append(np.array(x, dtype=float))
        ty.append(y[0])
    tx = np.array(tx)
    ty = np.array(ty)
    return tx, ty
    
def load_sample(name):
    with open("../dataset/features/" + name, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=",")
        x = []
        y = []
        for row in r:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x, dtype=float), np.array(y)
def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):
    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask
def weighted_accuracy(y_true, y_pred):
    return np.sum((np.array(y_pred).ravel() == np.array(y_true).ravel()))*1.0/len(y_true)


def unweighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies[num] = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])
    return np.mean(classes_accuracies)
   
def to_categorical(y):
    return label_binarize(y,emotions)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    shift = 2
    y_pred = y_pred[:, shift:, :]
    input_length -= shift
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(nb_feat, nb_class, optimizer='Adadelta'):
    net_input = Input(name="the_input", shape=(200, nb_feat))
    forward_lstm1  = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh"
                         )(net_input)
    backward_lstm1 = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh",
                          go_backwards=True
                         )(net_input)
    blstm_output1  = Merge(mode='concat')([forward_lstm1, backward_lstm1])

    forward_lstm2  = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh"
                         )(blstm_output1)
    backward_lstm2 = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh",
                          go_backwards=True
                         )(blstm_output1)
    blstm_output2  = Merge(mode='concat')([forward_lstm2, backward_lstm2])

    hidden = TimeDistributed(Dense(512, activation='tanh'))(blstm_output2)
    output = TimeDistributed(Dense(nb_class + 1, activation='softmax'))(hidden)

    labels = Input(name='the_labels', shape=[1], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([output, labels, input_length, label_length])

    model = Model(input=[net_input, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=[])

    test_func = K.function([net_input], [output])
    
    return model, test_func
model, test_func = build_model(nb_feat=nb_feat, nb_class=nb_class, optimizer=optimizer)
model.summary()
# data preparation 

X, y = get_sample()
y = np.argmax(to_categorical(y), axis=1)
y = np.reshape(y, (y.shape[0], 1))

X, X_mask = pad_sequence_into_array(X, maxlen=200)
y, y_mask = pad_sequence_into_array(y, maxlen=1)


index_to_retain = np.sum(X_mask, axis=1, dtype=np.int32) > 5

X, X_mask = X[index_to_retain], X_mask[index_to_retain]
y, y_mask = y[index_to_retain], y_mask[index_to_retain]


idxs_train, idxs_test = train_test_split(range(X.shape[0]))
X_train, X_test = X[idxs_train], X[idxs_test]
X_train_mask, X_test_mask = X_mask[idxs_train], X_mask[idxs_test]
y_train, y_test = y[idxs_train], y[idxs_test]
y_train_mask, y_test_mask = y_mask[idxs_train], y_mask[idxs_test]


# training 
sess = tf.Session()

class_weights = np.unique(y, return_counts=True)[1]*1.
class_weights = np.sum(class_weights) / class_weights

sample_weight = np.zeros(y_train.shape[0])
# for num, i in enumerate(y_train):
#     sample_weight[num] = class_weights[i[0]]

ua_train = np.zeros(nb_epoch)
ua_test = np.zeros(nb_epoch)
wa_train = np.zeros(nb_epoch)
wa_test = np.zeros(nb_epoch)
loss_train = np.zeros(nb_epoch)
loss_test = np.zeros(nb_epoch)

for epoch in range(nb_epoch):
    epoch_time0 = time.time()
    total_ctcloss = 0.0
    batches = range(0, X_train.shape[0], batch_size)
    shuffle = np.random.choice(batches, size=len(batches), replace=False)
    for num, i in enumerate(shuffle):
        inputs_train = {'the_input': X_train[i:i+batch_size],
                        'the_labels': y_train[i:i+batch_size],
                        'input_length': np.sum(X_train_mask[i:i+batch_size], axis=1, dtype=np.int32),
                        'label_length': np.squeeze(y_train_mask[i:i+batch_size]),
                       }
        
        outputs_train = {'ctc': np.zeros([inputs_train["the_labels"].shape[0]])}

        ctcloss = model.train_on_batch(x=inputs_train, y=outputs_train)

        total_ctcloss += ctcloss * inputs_train["the_input"].shape[0] * 1.
        print("Iteration:" + str(num) + ": loss " + str(ctcloss) )
    loss_train[epoch] = total_ctcloss / X_train.shape[0]

    inputs_train = {'the_input': X_train,
                    'the_labels': y_train,
                    'input_length': np.sum(X_train_mask, axis=1, dtype=np.int32),
                    'label_length': np.squeeze(y_train_mask),
                   }
    outputs_train = {'ctc': np.zeros([y_train.shape[0]])}
    preds = test_func([inputs_train["the_input"]])[0]
    decode_function = K.ctc_decode(preds[:,2:,:], inputs_train["input_length"]-2, greedy=False, top_paths=1)
    labellings = decode_function[0][0].eval(session=sess)
    if labellings.shape[1] == 0:
        ua_train[epoch] = 0.0
        wa_train[epoch] = 0.0
    else:
        ua_train[epoch] = unweighted_accuracy(y_train.ravel(), labellings.T[0].ravel())
        wa_train[epoch] = weighted_accuracy(y_train.ravel(), labellings.T[0].ravel())


    inputs_test = {'the_input': X_test,
                   'the_labels': y_test,
                   'input_length': np.sum(X_test_mask, axis=1, dtype=np.int32),
                   'label_length': np.squeeze(y_test_mask),
                  }
    outputs_test = {'ctc': np.zeros([y_test.shape[0]])}
    preds = test_func([inputs_test["the_input"]])[0]
    decode_function = K.ctc_decode(preds[:,2:,:], inputs_test["input_length"]-2, greedy=False, top_paths=1)
    labellings = decode_function[0][0].eval(session=sess)
    if labellings.shape[1] == 0:
        ua_test[epoch] = 0.0
        wa_test[epoch] = 0.0
    else:
        ua_test[epoch] = unweighted_accuracy(y_test.ravel(), labellings.T[0].ravel())
        wa_test[epoch] = weighted_accuracy(y_test.ravel(), labellings.T[0].ravel())
    loss_test[epoch] = np.mean(model.predict(inputs_test))

    epoch_time1 = time.time()


    print('epoch = %d, \
WA Tr = %0.2f, UA Tr = %0.2f, WA Te = %0.2f, UA Te = %0.2f, CTC Tr = %0.2f, CTC Te = %0.2f, \
time = %0.2fmins' % (epoch + 1, 
                     wa_train[epoch], ua_train[epoch], 
                     wa_test[epoch], ua_test[epoch], 
                     loss_train[epoch], loss_test[epoch],
                     (epoch_time1-epoch_time0)/60))
model_json = model.to_json()
with open("ctc_model_update.json", "w") as json_file:
  json_file.write(model_json)
model.save_weights("ctc_weights_update.h5")





