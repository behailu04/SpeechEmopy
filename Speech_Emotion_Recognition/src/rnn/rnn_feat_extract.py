import os
import librosa 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_class_names ( path="rnn_features_pos_neu"):
    class_names = os.listdir(path)
    return class_names

def encode_class(class_name, class_names):
    index = class_names.index(class_name)
    vector = np.zeros(len(class_names))
    vector[index] = 1
    return vector

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size //2)
def merge_shuffle_file(input_path = "rnn_d/"):
    t_all_audio_files = []
    t_all_labels = []
    tst_all_audio_files = []
    tst_all_labels = []
    class_names = os.listdir(input_path)
    for index1, class_name in enumerate(class_names):
        train_class_files = os.listdir(input_path + class_name + "/train/")
        test_class_files = os.listdir(input_path + class_name + "/test/")
        # t_all_audio_files = np.concatenate((t_all_audio_files, train_class_files))
        # tst_all_audio_files = np.concatenate((tst_all_audio_files, test_class_files))
        
        for i, train_file in enumerate(train_class_files):
            print ("Train Class: " + class_name  + " File: " + input_path + class_name + "/train/" + train_file)
            t_all_audio_files.append(input_path + class_name + "/train/" + train_file)
            t_all_labels.append(class_name)
        for j, test_file in enumerate(test_class_files):
            print ("Test Class: " + class_name  + " File: " + input_path + class_name + "/test/" + test_file)
            tst_all_audio_files.append(input_path + class_name + "/test/" + test_file)
            tst_all_labels.append(class_name)
    t_all_audio_files = np.asarray(t_all_audio_files)
    t_all_labels = np.asarray(t_all_labels)
    tst_all_audio_files = np.asarray(tst_all_audio_files)
    tst_all_labels = np.asarray(tst_all_labels)
    assert len(t_all_audio_files) == len(t_all_labels)
    p = np.random.permutation(len(t_all_audio_files))
    assert len(tst_all_audio_files) == len(tst_all_labels)
    q = np.random.permutation(len(tst_all_audio_files))
    return t_all_audio_files[p], t_all_labels[p], tst_all_audio_files[q], tst_all_labels[q]

def extract_features(input_path = "rnn_d/", output_path = "rnn_features_pos_neu/", bands = 20, frames = 41):
    window_size = 512 * (frames -1)
    train_mfccs = []
    test_mfccs = []
    train_labels = []
    test_labels = []
    train_audio_files, train_class_labels, test_audio_files, test_audio_labels = merge_shuffle_file()
    class_names = os.listdir(input_path)
    for index1, train_file_path in enumerate(train_audio_files):
        print("processing: " + train_file_path)
        audio_clip, sr = librosa.load(train_file_path)
        for (start,end) in windows(audio_clip,window_size):
            if(len(audio_clip[start:end]) == window_size):
                signal = audio_clip[start:end]
                mfcc = librosa.feature.mfcc(y = signal, sr = sr, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                train_mfccs.append(mfcc)
                train_labels.append(encode_class(train_class_labels[index1], class_names))
    for index2, test_file_path in enumerate(test_audio_files):
        print("processing: " + test_file_path)
        audio_clip, sr = librosa.load(test_file_path)
        for (start, end) in windows(audio_clip,window_size):
            if(len(audio_clip[start:end]) == window_size):
                signal = audio_clip[start:end]
                mfcc = librosa.feature.mfcc(y = signal, sr = sr, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                test_mfccs.append(mfcc)
                test_labels.append(encode_class(test_audio_labels[index2], class_names))
        
    train_features = np.asarray(train_mfccs).reshape(len(train_mfccs), bands, frames)
    test_features = np.asarray(test_mfccs).reshape(len(test_mfccs), bands, frames)
    return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels)

def pre_process():
    output_path = "rnn_features_pos_neu/"
    X_train, Y_train, X_test, Y_test = extract_features()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + "train/"):
        os.mkdir(output_path + "train/")
    if not os.path.exists(output_path + "test/"):
        os.mkdir(output_path + "test/")
    train_feature_file = output_path + "train/" + "train_x.npy"
    test_feature_file = output_path + "test/" + "test_x.npy"
    train_label_file = output_path + "train/" + "train_y.npy"
    test_label_file = output_path + "test/" + "test_y.npy"

    np.save(train_feature_file, X_train)
    print("Saved: " + train_feature_file)
    np.save(train_label_file, Y_train)
    print("Saved: " + train_label_file)

    np.save(test_feature_file, X_test)
    print("Saved: "  + test_feature_file)
    np.save(test_label_file, Y_test)
    print("Saved: " + test_label_file)
pre_process()


    