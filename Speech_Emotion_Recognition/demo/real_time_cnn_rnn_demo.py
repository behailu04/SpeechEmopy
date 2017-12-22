import Tkinter as tk
from Tkinter import *
from PIL import ImageTk, Image
import numpy as np
import matplotlib, sys
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg , NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from random import randrange
from scipy.io import wavfile
import pyaudio
import wave
# from recorder import *
from Model import *
from threading import Thread

class Demo(object):
    def __init__(self, master, **kwargs):
        self.qu = queue.Queue()
        self.RATE = 16000
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.CHUNKS = 2**14
        global img
        self.master=master
        self.isRecording = False
        # self.recorder = Recorder(channels=1)
        pad=3
        self.model = Model()
        self._geom='200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        master.bind('<Escape>',self.toggle_geom)
        f = Frame(master, bg="#00111a")
        f.pack()
        title = Label(f, text="Speech Emotion Classifier", fg="#33bbff", bg="#00111a", pady=20)
        title.config(font=("Courier", 40))
        title.pack()
        im = Image.open('mic.png')
        re = im.resize((70, 70),Image.ANTIALIAS)
        img = ImageTk.PhotoImage(re)
        self.status = Label(f,text="Record", fg="white", bg="#00111a")
        self.status.pack()
        # self.reorder = Recorder()
        # img = PhotoImage(file="mic.gif")
        mic = tk.Button(f, borderwidth=0, command=self.rec)
        mic.configure(state = "active",  bg="#00111a")
        mic.bind("Enter", lambda event, h=mic: h.configure(bg="#00111a"))
        mic.bind("<Leave>", lambda event, h=mic: h.configure(bg="#00111a"))
        mic.configure(state = "normal", bg="#00111a")
        mic.config(image=img)
        mic.pack()
        
        frame = Frame(master,bg="#00111a")
        frame.pack(side=LEFT)
        self.result = Label(frame, text="Result", fg="white", bg="#00111a")
        self.result.pack()
        fig = Figure(figsize=(5, 4), dpi=100,facecolor='#00111a')
        
        a = fig.add_subplot(211)
        a.set_facecolor('#00111a') 
        t = arange(0.0, 0.3, 0.01)
        s = sin(2*pi*t)
        sf, self.signal = wavfile.read("happy.wav")
        a.plot(self.signal)
        
        dataPlot = FigureCanvasTkAgg(fig, master=frame)
        
        dataPlot.show()
        dataPlot.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=1)
        
        

    def rec(self):
        if(self.isRecording):
            self.isRecording = False
            self.status.configure(text="Record")
            self.stream.stop_stream()
            self.stream.close()
            # self.reco.stop_recording()
            # self.result.configure(text= self.model.cnn_predict("test/happy1.wav"))
            
        else:
            # self.start_record()
            self.start()
            self.isRecording = True
            self.status.configure(text="Recording...")
            # with self.recorder.open("new.wav", 'wb') as recfile:
            #     self.reco = recfile
            #     # self.reco.start_recording()
        
        print("Clicked")
    def start_recording():
        pass
    def stop_recording():
        pass
    def toggle_geom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom
    def start_record(self):
        py = pyaudio.PyAudio()
        self.stream = py.open(format=self.FORMAT, channels=self.CHANNELS, rate= self.RATE, input=True, frames_per_buffer=self.CHUNKS)
        while(self.stream.is_active):
            self.data = self.stream.read(self.CHUNKS)
            self.data = np.fromstring(self.data, np.int16)
            self.qu.put(self.data)
        self.stream.stop_stream()
        self.stream.close()
        py.terminate()
    def start_classify(self):
       while True:
           if not self.qu.empty():
                item = self.qu.get()
                print()
                result = self.model.rnn_predict_feat(item)
                print(result)
                self.result.configure(text=result )
    def start(self):
        new_thread=Thread(target=self.start_record)
        new_thread.start()
        new_thread2=Thread(target=self.start_classify)
        new_thread2.start()
root = tk.Tk()
root.configure(background='#00111a')
app=Demo(root)
root.mainloop()