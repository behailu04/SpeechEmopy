import os
import shutil
import sys
import random


def split_train_test(input_path = "data/", output_path = "dataset/", train_percentage = 0.8):
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	class_names = os.listdir(input_path)
	for index, class_name in enumerate(class_names):
		out1 = output_path + class_name + "/test/"
		out2 = output_path + class_name + "/train/"
		inp = input_path + class_name + "/"
		
		if not os.path.exists(out1):
			os.mkdir(out1)
		if not os.path.exists(out2):
			os.mkdir(out2)
		class_files = os.listdir(input_path + class_name)
		random.shuffle(class_files)
		n_files = len(class_files)
		n_train = n_files * train_percentage
		for index2, filename in enumerate(class_files):
			if index2 < n_train:
				shutil.move(inp + filename, out2 + filename)
			else:
				shutil.move(inp + filename, out1 + filename)

		print("Class: " + class_name + " has: " + str(n_files))
split_train_test()
# random.shuffle(filse)

# des_fold = "way/"
# des_fold2 = "w/"
# if not os.path.exists(des_fold):
# 	os.mkdir(des_fold)
# if not os.path.exists(des_fold2):
# 	os.mkdir(des_fold2)

# siz = len(filse)
# train = siz * 0.5
# for i, file in enumerate(filse):
# 	path2 = des_fold + file
# 	path3 = des_fold2 + file
# 	if i < train:
# 		shutil.move(path + file, path2)
# 	else:
# 		shutil.move(path + file, path3)

