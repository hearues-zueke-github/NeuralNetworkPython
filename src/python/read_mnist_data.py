#! /usr/bin/python3.10.2

import gzip
import os
import requests
import string
import subprocess
import sys
import time
import yaml

import multiprocessing as mp
import numpy as np
import pandas as pd

from io import StringIO
from memory_tempfile import MemoryTempfile

from numpy.random import Generator, PCG64
from scipy.sparse import csr_matrix, coo_matrix

from PIL import Image
from typing import Dict

from utils_platform import get_processor_name

HOME_DIR = os.path.expanduser("~")
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PROJECT_PATH_DIR = os.path.join(PATH_ROOT_DIR, '../..')
TEMP_DIR = MemoryTempfile().gettempdir()

MNIST_DATA_DIR_PATH = os.path.join(MAIN_PROJECT_PATH_DIR, "mnist_data")

def download_files():
	l_file_name_gz_file_name = [
		("train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"),
		("train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"),
		("t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"),
		("t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte"),
	]

	def download_file_and_extract(url, file_path_gz, file_path):
		r = requests.get(url)
		with open(file_path_gz, 'wb') as f:
			f.write(r.content)

		with gzip.open(file_path_gz, 'rb') as f_in:
			with open(file_path, 'wb') as f_out:
				f_out.write(f_in.read())

	base_url = 'http://yann.lecun.com/exdb/mnist/'
	for file_name_gz, file_name in l_file_name_gz_file_name:
		file_path = os.path.join(MNIST_DATA_DIR_PATH, file_name)
		if not os.path.exists(file_path):
			file_path_gz = os.path.join(MNIST_DATA_DIR_PATH, file_name_gz)
			download_file_and_extract(url=base_url+file_name_gz, file_path_gz=file_path_gz, file_path=file_path)


def extract_train_test_data():
	file_path_train_images = os.path.join(MNIST_DATA_DIR_PATH, "train-images.idx3-ubyte")
	file_path_train_labels = os.path.join(MNIST_DATA_DIR_PATH, "train-labels.idx1-ubyte")
	file_path_test_images = os.path.join(MNIST_DATA_DIR_PATH, "t10k-images.idx3-ubyte")
	file_path_test_labels = os.path.join(MNIST_DATA_DIR_PATH, "t10k-labels.idx1-ubyte")

	assert os.path.exists(file_path_train_images)
	assert os.path.exists(file_path_train_labels)
	assert os.path.exists(file_path_test_images)
	assert os.path.exists(file_path_test_labels)

	with open(file_path_train_images, 'rb') as f:
		arr_info_big = np.fromfile(file=f, dtype=np.int8, count=4*4)
		arr_info = np.ndarray(shape=(4, ), dtype='>i4', buffer=arr_info_big)
		assert arr_info[0] == 0x00000803
		amount = arr_info[1]
		rows = arr_info[2]
		cols = arr_info[3]

		arr_train_images = np.fromfile(file=f, dtype=np.uint8, count=amount*rows*cols)

	with open(file_path_train_labels, 'rb') as f:
		arr_info_big = np.fromfile(file=f, dtype=np.int8, count=4*4)
		arr_info = np.ndarray(shape=(4, ), dtype='>i4', buffer=arr_info_big)
		assert arr_info[0] == 0x00000801
		amount = arr_info[1]

		arr_train_labels = np.fromfile(file=f, dtype=np.uint8, count=amount)

	with open(file_path_test_images, 'rb') as f:
		arr_info_big = np.fromfile(file=f, dtype=np.int8, count=4*4)
		arr_info = np.ndarray(shape=(4, ), dtype='>i4', buffer=arr_info_big)
		assert arr_info[0] == 0x00000803
		amount = arr_info[1]
		rows = arr_info[2]
		cols = arr_info[3]

		arr_test_images = np.fromfile(file=f, dtype=np.uint8, count=amount*rows*cols)

	with open(file_path_test_labels, 'rb') as f:
		arr_info_big = np.fromfile(file=f, dtype=np.int8, count=4*4)
		arr_info = np.ndarray(shape=(4, ), dtype='>i4', buffer=arr_info_big)
		assert arr_info[0] == 0x00000801
		amount = arr_info[1]

		arr_test_labels = np.fromfile(file=f, dtype=np.uint8, count=amount)

	d_data = {
		'arr_train_images': arr_train_images,
		'arr_train_labels': arr_train_labels,
		'arr_test_images': arr_test_images,
		'arr_test_labels': arr_test_labels,
	}

	return d_data


def create_images_from_data(d_data):
	file_path_img_train = os.path.join(MNIST_DATA_DIR_PATH, 'img_train.png')
	file_path_img_test = os.path.join(MNIST_DATA_DIR_PATH, 'img_test.png')

	if not os.path.exists(file_path_img_train):
		arr_train_image = d_data['arr_train_images'].reshape((60000, 28, 28)).transpose(0, 2, 1).reshape((600, 100*28, 28)).transpose(0, 2, 1).reshape((600*28, 100*28))
		img_train = Image.fromarray(arr_train_image)
		img_train.save(file_path_img_train)

	if not os.path.exists(file_path_img_test):
		arr_test_image = d_data['arr_test_images'].reshape((10000, 28, 28)).transpose(0, 2, 1).reshape((100, 100*28, 28)).transpose(0, 2, 1).reshape((100*28, 100*28))
		img_test = Image.fromarray(arr_test_image)
		img_test.save(file_path_img_test)


if __name__ == '__main__':
	print('Download files if needed.')
	download_files()

	print('Extract the train and test data from the files.')
	d_data = extract_train_test_data()

	print('Create images from the data.')
	create_images_from_data(d_data)
