import Tkinter as tk
import wave
import time
from pydub import AudioSegment
import pyaudio
import numpy
import audioSegmentation as aS
import os
import audioBasicIO as aIO
import threading
from Queue import Queue
import random
from Model import *
import ui_plot
import sys
import numpy
from PyQt4 import QtCore, QtGui
import PyQt4.Qwt5 as Qwt



class App():
    def __init__(self, master):
        self.BUF_SIZE = 100
        self.numPoints = 1000
        self.xs = numpy.arange(self.numPoints)
        self.ys = numpy.sin(3.14159 * self.xs * 10 / self.numPoints)
        self.startlen = 0
        self.endlenn = 0
        self.frames = []
        self.fullFrame = []
        self.segment = []
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.p = pyaudio.PyAudio()
        self.interval = int((self.RATE / self.CHUNK * 5))
        self.isrecording = False
        self.button = tk.Button(main, text='rec')
        self.button.bind("<Button-1>", self.startrecording)
        self.button.bind("<ButtonRelease-1>", self.stoprecording)
        self.button.pack()
        self.model = Model()
    def startrecording(self, event):
        self.isrecording = True
        t = threading.Thread(target=self._record)
        t.start()

    def stoprecording(self, event):
        self.isrecording = False

    def plotSomething(self):
        global ys
        ys = numpy.roll(self.ys, -1)
        print "PLOTTING"
        self.c.setData(self.xs, ys)
        ui_plot.qwtPlot.replot()
    def _record(self):
        print("please speak a word into the microphone")
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.RECORD_SECONDS = 6
        self.count = 0
        self.current = 0
        self.cutinterval = 105
        self.segment = []
        self.fullFrame=[]
        print("saving and segmenting")

        while(self.isrecording):
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
            self.fullFrames=self.frames
            print(len(self.fullFrames))
            self.path = "Record/recordedSegment" + str(int(time.time())) + ".wav"
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.segment=self.frames[self.count:len(self.frames)]
            self.count = self.count + self.cutinterval
            wf = wave.open(self.path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.segment))
            wf.close()
            print("segment Recording Finished")
            if(len(self.segment)>100 ):
                audiofile = AudioSegment.from_wav(self.path)
                [Fs, x] = aIO.readAudioFile(self.path)
                segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow=1.0, Weight=0.3, plot=False)
                newname = time.time()
                for i in range(0, len(segments)):
                    z = segments[i]
                    startpoint = z[0]
                    endpoint = z[1]
                    sliceSound = audiofile[startpoint * 1000: endpoint * 1000]
                    sliceSound.export(self.path, format="wav")
                    newname = newname + 1
                    print("--Segmenter finished--")
                result = self.model.cnn_predict(self.path)
                print("predicting......")
                print(result)
                print("done.....")

        if (len(self.frames) != 0 & self.isrecording == False):
            print(len(self.frames))
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.paths = "Record/FullRecord" + str(int(time.time())) + ".wav"
            wf = wave.open(self.paths, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.fullFrames))
            wf.close()
            self.frames=[]
            self.segment = []
            print("--Finished Recording--")
if __name__ == '__main__':
    main = tk.Tk()
    main.title("Emotion Recoginition From Speech")
    main.minsize(width=300, height=300)
    app = App(main)
    main.mainloop()
