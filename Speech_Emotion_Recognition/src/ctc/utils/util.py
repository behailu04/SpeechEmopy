import os
import csv
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

class util:
    def __init__(self):
        real_path = os.path.dirname(os.path.realpath(__file__))
        self.emotions = np.array(["Happy", "Sad", "Surprise", "Fear", "Angry", "Neutral"])
        self.path_to_data = real_path + "/../../dataset/sessions/"
        self.path_to_features = real_path + "/../../dataset/features/"
        self.sessions = ["Session1", "Session2", "Session3", "Session4", 'Session5']
    def get_emotion_name(self,emo):
        emos = {'exc':'Happy', 'neu':'Neutral', 'xxx':'XXX','sad':'Sad', 'fru':'Frustration',
        'ang':'Angry', 'hap':'Happy', 'sur':'Surprise','fea':'Fear','dis':'Disguest', 'oth':'XXX'}
        return emos[emo]
    def read_iemocap_dataset(self):
        data = []
        for session in self.sessions:
            path_to_wav = self.path_to_data + session + '/dialog/wav/'
            path_to_emotions = self.path_to_data + session + '/dialog/EmoEvaluation/'
            files = os.listdir(path_to_wav)
            files = [f[:-4] for f in files if f.endswith(".wav")]
            print("Session:" + str(session) + " has " + str(len(files)) + " files")
            for f in files:
                print("Processing " + f)
                samplerate, wav = self.get_audio(path_to_wav, f + ".wav")
                emotions = self.get_emotions(path_to_emotions, f + '.txt')
                sample = self.split_audio(samplerate,wav, emotions)
                for i , emo in enumerate(emotions):
                    emo['signal'] = sample[i]['signal']
                    if (emo['emotion'] in self.emotions):
                        data.append(emo)
            print(session, "Completed")
        sort_key = self.get_field(data, 'id')
        return np.array(data)[np.argsort(sort_key)]



    def get_field(self,data, key):
        return np.array([e[key] for e in data])
    def get_emotions(self, path_to_emotions, name):
        file = open(path_to_emotions + name, 'r').read()
        file = np.array(file.split('\n'))
        idx = file == ''
        indx_n = np.arange(len(file))[idx]
        emotion = []
        for i in range(len(indx_n) - 2):
            g = file[indx_n[i] + 1:indx_n[i + 1]]
            head = g[0]
            i0 = head.find(' - ')
            start_time = float(head[head.find('[') + 1:head.find(' - ')])
            end_time = float(head[head.find(' - ') + 3:head.find(']')])
            actor_id = head[head.find(name[:-4]) + len(name[:-4]) + 1:
                        head.find(name[:-4]) + len(name[:-4]) + 5]
            emo = head[head.find('\t[') - 3:head.find('\t[')]
            emo = self.get_emotion_name(emo) 
            emotion.append({'start':start_time,
                            'end': end_time,
                            'id':name[:-4] + '_' + actor_id,
                            'emotion':emo})    
        return emotion
    def split_audio(self,samplerate, signal, emotions):
        frames = []
        for i, emo in enumerate(emotions):
            start = emo['start']
            end = emo['end']
            emo['signal'] = signal[int(start * samplerate): int(end * samplerate)]
            frames.append({'signal':emo['signal']})
        return frames
    def get_features(self, data):
        print(len(data))
        for i , d in enumerate(data):
            
            print("Extracting feature", d['id'])
            features = mfcc(signal = d['signal'], winlen=0.2, winstep=0.1, nfft=3200)
            x = []
            y = []
            for feat in features:
                x.append(feat)
                y.append(d['emotion'])
            x = np.array(x, dtype=float)
            y = np.array(y)
            self.save_sample(x,y, self.path_to_features + d['id'] + '.csv')

    def save_sample(self,x, y, name):
        with open(name, 'w') as csvfile:
            w = csv.writer(csvfile, delimiter=',')
            for i in range(x.shape[0]):
                row = x[i, :].tolist()
                row.append(y[i])
                w.writerow(row)
    

    def load_sample():
        pass
    def read_samples():
        pass
    def get_audio(self, path_to_wav, name):
        (samplerate, audio) = wav.read(path_to_wav + name)
        return samplerate, audio

        
u= util()
data = u.read_iemocap_dataset()
u.get_features(data)
