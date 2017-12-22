from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
class CTC:
    def __int__(self):
        self.model = self.load_model()
        self.session = tf.Session()
    def get_audio(self, path_to_wav, name):
        (samplerate, audio) = wav.read(path_to_wav + name)
        return samplerate, audio
    def prepare_data(self):
        sr, signal = self.get_audio("test/", "Ses02F_impro03.wav")
        features = mfcc(signal = signal, winlen=0.2, winstep=0.1, nfft=3200)
        return features
    def pad_sequence_into_array(self,Xs, maxlen=200, truncating='post', padding='post', value=0.):
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
    def load_model(self):
        graph = "ctc_model_update.json"
        weight = "ctc_weights_update.h5"
        json_file = open(graph, 'r')
        loaded_model_json = json_file.read()
        json_file.close
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weight)
        return loaded_model
    def predict(self, data, input_length):
        self.model = self.load_model()
        input_layer = self.model.layers[0].input
        output_layer = self.model.layers[-5].output
        decoder = K.function([input_layer, K.learning_phase()], [output_layer]) 
        preds = decoder([data])[0]
        self.session = tf.Session()
        decode_function = K.ctc_decode(preds[:,2:,:], input_length-2, greedy=False, top_paths=1 )
        labllings = decode_function[0][0].eval(session=self.session)
        sequences = labllings.T[0].ravel()
        return sequences
        # for i in range(len(sequences)):
        #     print(self.get_emotion[sequences[i]])
        #print(labllings.T[0].ravel())
    def get_emotion(self,index):
        emotions = {0:"Happy", 1:"Sad", 2:"Surprise", 3:"Fear", 4:"Angry", 5:"Neutral", -1:"b"}
        return emotions[index]
c = CTC()
features = c.prepare_data()
feat = []
print(features.shape[0]/200)
j = 0;
for i in range(features.shape[0]/200):
    feat.append(features[j:j+200, :])
    j = j + 200
feat = np.array(feat)
print(feat.shape)
x, x_mask = c.pad_sequence_into_array(feat)
sequ = c.predict(x, np.sum(x_mask, axis=1, dtype=np.int32))
sequ = sequ.tolist()
s = []
for k, j in enumerate(sequ):
    if(j != -1):
        s.append(j)
result = ""
for i , index in enumerate(s):
    result = result + " : "
    result += c.get_emotion(s[index])
print(result)
# print(sequ.shape)
# x, x_mask = c.pad_sequence_into_array(feat[200:400])
# c.predict(x, np.sum(x_mask, axis=1, dtype=np.int32))
# x, x_mask = c.pad_sequence_into_array(feat[400:600])
# c.predict(x, np.sum(x_mask, axis=1, dtype=np.int32))
# print(c.get_emotion(0))
    
        
