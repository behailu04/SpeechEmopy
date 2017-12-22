import numpy as np
from keras.models import model_from_json
import os
import librosa
EMOTIONS = ['fear', 'surprise','neutral','angry','sad','happy'
]


def load_model():
    json_file = open('model/rnn4_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model/rnn4_model.h5')
    return loaded_model
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features_array(filename, bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    sound_clip,s = librosa.load(filename)        
    for (start,end) in windows(sound_clip,window_size):
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
def extract_rnn_features_array(filename, bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    sound_clip,s = librosa.load(filename)
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
            mfccs.append(mfcc)
            
    features = np.asarray(mfccs).reshape(len(mfccs),bands,frames)
    return np.array(features)
def score(predictions):
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
    return EMOTIONS[index]
def predict(file, actual):
    feature_x = extract_features_array(file, bands = 20, frames = 41)
    predictions = model.predict(feature_x)
    # score(predictions)
    # predictions = prediction[0]
    # ind = np.argpartition(predictions[0], -2)[-2:]
    # ind[np.argsort(predictions[0][ind])]
    
    # ind = ind[::-1]
    # print "Actual:", actual, " Top guess: ", EMOTIONS[ind[0]], " (",round(predictions[0,ind[0]],3),")"
    # print "2nd guess: ", EMOTIONS[ind[1]], " (",round(predictions[0,ind[1]],3),")"
    # print(predictions)
    # index = np.argmax(predictions)
    print("-------------------------------------------------------")
    print("Actual:" + actual + ",Predicted:" + score(predictions))
    print("=======================================================\n")

def main():
    predict("test_data/sad/sad1.wav", "Sad")
    predict("test_data/sad/sad2.wav", "Sad")
    predict("test_data/happy/happy1.wav", "Happy")
    predict("test_data/happy/happy2.wav", "Happy")
    predict("test_data/fear/fear1.wav", "Fear")
    predict("test_data/fear/fear2.wav", "Fear")
    predict("test_data/surprise/sur1.wav", "Surprise")
    predict("test_data/surprise/sur2.wav", "Surprise")
    predict("test_data/neutral/neu1.wav", "Neutral")
    predict("test_data/neutral/neu2.wav", "Neutral")
    predict("test_data/angry/angry1.wav", "Angry")
    predict("test_data/angry/angry2.wav", "Angry")
model = load_model()
main()


# mlp = MlP()
# mlp.predict("filepath.wav")