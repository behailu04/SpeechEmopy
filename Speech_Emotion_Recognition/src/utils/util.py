import os
import numpy as np
import scipy.io.wavfile as wav
class Util:
    def __init__(self, path):
        self.path = path
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    def get_emotions(self, path_to_emotion, filename):
        file = open(path_to_emotion + filename, 'r').read()
        file = np.array((file.split('\n')))
        idx = file == ''
        idx_n = np.arange(len(file))[idx]
        emotions = []
        for i in range(len(idx_n) - 2):
            emo_info = file[idx_n[i] + 1:idx_n[i + 1]]
            head = emo_info[0]
            i0 = head.find(' - ')
            start_time = float(head[head.find('[') + 1: head.find(' - ')])
            end_time = float(head[head.find(' - ') + 3:head.find(']')])
            actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:head.find(filename[:-4]) + len(filename[:-4]) + 5]
            emo = head[head.find('\t[') - 3:head.find('\t[')]
            emo = self.get_emotion_name(emo)
            emotions.append({'start':start_time,
                            'end':end_time,
                            'id':filename[:-4] + '_' + actor_id,
                            'emotion':emo})
        return emotions
    def get_emotion_name(self,emo):
        emos = {'exc':'Excited', 'neu':'Neutral', 'xxx':'XXX','sad':'Sad', 'fru':'Frustration',
        'ang':'Angry', 'hap':'Happy', 'sur':'Surprise','fea':'Fear','dis':'Disguest', 'oth':'XXX'}
        return emos[emo]
    def read_iemocap_database(self):
        data = []
        samplerate = 0
        for session in self.sessions:
             path_to_wav = self.path + session + '/dialog/wav/'
             path_to_emotions = self.path + session + '/dialog/EmoEvaluation/'
             files = os.listdir(path_to_wav)
             files = [f[:-4] for f in files if f.endswith('.wav')]
             for f in files:
                 samplerate, wav = self.get_audio(path_to_wav, f + '.wav')
                 emotions = self.get_emotions(path_to_emotions, f + '.txt')
                 samples = self.segment_audio(samplerate, wav, emotions)
                 for i , emo in enumerate(emotions):
                     emo['signal'] = samples[i]['signal']
                     data.append(emo)
        return data, samplerate
    def get_audio(self, path_to_wav, file):
        (samplerate, audio) = wav.read(path_to_wav + file)
        return samplerate, audio
    def segment_audio(self, samplerate, signal, emotions):
        frames = []
        print("Spliting audio")
        for i , emo in enumerate(emotions):
            print(emo['id'])
            start = emo['start']
            end = emo['end']
            emo['signal'] = signal[int(start * samplerate): int(end * samplerate)]
            frames.append({'signal':emo['signal']})
        return frames
    def save_audio(self, path):
        datas, samplerate = self.read_iemocap_database()
        print("Writing files")
        for i, data in enumerate(datas):
            filename = data['emotion'] + '_' + data['id'] + '.wav'
            if not os.path.exists( path):
                os.mkdir(path, 0755)
            if not os.path.exists( path  + data['emotion']):
                os.mkdir(path  + data['emotion'], 0755)
            wav.write(path + data['emotion'] + '/' + filename, samplerate, data['signal'])
            print(filename)
    def main(self):
        path = '../../datasets/corpus/IEMOCAP_LABELED_DATABASE/'
        self.save_audio(path)
util = Util('../../datasets/corpus/IMEOCAP/sessions/')
util.main()

        
