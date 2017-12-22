import soundfile as sf
import pyrubberband as pyrb
import os
import scipy.io.wavfile as wav

class AudioAgumentation:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
    def do_agumentation(self):
        no_class = os.listdir(self.input_path)
        for name in no_class:
            files = os.listdir(self.input_path  + name + "/")
            for i, audio in enumerate(files):
                y, sr = sf.read(self.input_path + name + "/" + audio)
                y_strech = pyrb.time_stretch(y, sr, 2.0)
                wav.write(self.output_path + name, sr, y_strech)
                print(name, "has augmented and saved")

a = AudioAgumentation("../datasets/corpus/iemocap_ravdes_savee_pos_neu/", "../datasets/corpus/augmented/")
a.do_agumentation()
