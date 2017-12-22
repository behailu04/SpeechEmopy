import wave
import os


def count():
    total = 0
    path = "../datasets/corpus/iemocap_ravdes_savee_pos_neu/"
    # path = "../datasets/corpus/i/"
    classes = os.listdir(path)
    for i, c in enumerate(classes):
        n_files = os.listdir(path + c)
        for k, f in enumerate(n_files):
            total += get_duration(path + c + "/" + f)
    return total
def get_duration(path):
    wav = wave.open(path, mode='r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    duration = nframes / float(framerate)
    return duration
total = count()
print(total, "sec")
min = total //60
# print(min)
sec = total % 60
hour = min // 60
m = min % 60
print(str(int(hour)) + "hr," + str(int(m)) + "min," + str(int(sec)) + "sec")
