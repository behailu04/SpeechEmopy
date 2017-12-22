import numpy as np
from keras.models import model_from_json
import os
import librosa

class Model:
    def __init__(self):
        self.EMOTIONS = ['fear', 'surprise','neutral','angry','sad','happy']
        self.type = 0;
    def load_model(self):
        graph = "models/cnn/baseline/cnn_baseline_model.json"
        weight = "models/cnn/baseline/cnn_baseline_weights.h5"
        if(self.type ==1):
            graph = "models/rnn/baseline/rnn_baseline_model.json"
            weight = "models/rnn/baseline/rnn_baseline_weights.h5"
        json_file = open(graph, 'r')
        loaded_model_json = json_file.read()
        json_file.close
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weight)
        return loaded_model
    def windows(self,data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size / 2)
    def extract_cnn_features_array(self,filename, bands = 60, frames = 41):
        window_size = 512 * (frames - 1)
        log_specgrams = []
        sound_clip,s = librosa.load(filename)        
        for (start,end) in self.windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.logamplitude(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        return np.array(features)
    def extract_rnn_features_array(self,filename, bands = 20, frames = 41):
        window_size = 512 * (frames - 1)
        mfccs = []
        sound_clip,s = librosa.load(filename)
        for (start,end) in self.windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                mfccs.append(mfcc)
        # print(len(mfccs))
        features = np.asarray(mfccs).reshape(len(mfccs),bands,frames)
        return np.array(features)
    def score(self,predictions):
        fear = 0
        surprise = 0
        nutral = 0
        angry = 0
        sad = 0
        happy = 0
        for i in range(len(predictions)):
            fear += predictions[i][0]
            surprise += predictions[i][1]
            nutral += predictions[i][2]
            angry += predictions[i][3]
            sad += predictions[i][4]
            happy += predictions[i][5]
        score = [fear, surprise, nutral, angry, sad, happy]
        index = np.argmax(score)
        return self.EMOTIONS[index]
    def extract_rnn(self,filename, bands = 20, frames = 41):
        window_size = 512 * (frames - 1)
        mfccs = []
        sound_clip,s = filename, 16000
        for (start,end) in self.windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                mfccs.append(mfcc)
        # print(len(mfccs))
        features = np.asarray(mfccs).reshape(len(mfccs),bands,frames)
        return np.array(features)
    def rnn_predict(self,file):
        self.type = 1
        feature_x = self.extract_rnn_features_array(file, bands = 20, frames=41)
        model = self.load_model()
        predictions = model.predict(feature_x)
        return self.score(predictions)
    def rnn_predict_feat(self,file):
        self.type = 1
        feature_x = self.extract_rnn(file, bands = 20, frames=41)
        model = self.load_model()
        predictions = model.predict(feature_x)
        return self.score(predictions)
    def cnn_predict(self,file):
        self.type = 0
        feature_x = self.extract_cnn_features_array(file, bands = 60, frames = 41)
        model = self.load_model()
        # print(feature_x.shape)
        predictions = model.predict(feature_x)
        return self.score(predictions)
    def extract_cnn(self,filename, bands = 60, frames = 41):
        window_size = 512 * (frames - 1)
        log_specgrams = []
        sound_clip,s = filename, 16000       
        for (start,end) in self.windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.logamplitude(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        return np.array(features)
    def cnn_predict_feat(self,file):
        self.type = 0
        feature_x = self.extract_cnn(file, bands = 60, frames = 41)
        # print(feature_x.shape)
        model = self.load_model()
        predictions = model.predict(feature_x)
        return self.score(predictions)
model = Model()

# print("------------------------Happy----------------")
# print("CNN:1: "+model.cnn_predict("test/happy1.wav"))
# print("RNN:1: "+model.rnn_predict("test/test_happy1.wav"))
# print("CNN:2: "+model.cnn_predict("test/happy2.wav"))
# print("RNN:2: "+model.rnn_predict("test/test_happy2.wav"))
# print("----------------------------------------------\n\n")
# print("------------------------Angry----------------")
# print("CNN:1: "+model.cnn_predict("test/angry1.wav"))
# print("RNN:1: "+model.rnn_predict("test/test_ang.wav"))
# print("CNN:2: "+ model.cnn_predict("test/angry2.wav"))
# print("RNN:2: "+model.rnn_predict("test/test_ang2.wav"))
# print("----------------------------------------------\n\n")
# print("------------------------Sad----------------")
# print("CNN:1: "+model.cnn_predict("test/sad1.wav"))
# print("RNN:1: "+model.rnn_predict("test/sad1.wav"))
# print("CNN:2: "+model.cnn_predict("test/sad2.wav"))
# print("RNN:2: "+model.rnn_predict("test/sad2.wav"))
# print("----------------------------------------------\n\n")
# print("------------------------Fear----------------")
# print("CNN:1: "+model.cnn_predict("test/fear1.wav"))
# print("RNN:1: "+model.rnn_predict("test/fear1.wav"))
# print("CNN:2: "+model.cnn_predict("test/fear2.wav"))
# print("RNN:2: "+model.rnn_predict("test/fear2.wav"))
# print("----------------------------------------------\n\n")
#
# print("------------------------Surprise----------------")
# print("CNN:1: "+model.cnn_predict("test/sur1.wav"))
# print("RNN:1: "+model.rnn_predict("test/sur1.wav"))
# print("CNN:2: "+model.cnn_predict("test/sur2.wav"))
# print("RNN:2: "+model.rnn_predict("test/sur2.wav"))
# print("----------------------------------------------\n\n")
