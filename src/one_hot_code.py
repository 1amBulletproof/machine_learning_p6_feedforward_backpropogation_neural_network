#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	One hot code data

from file_manager import FileManager
from data_manipulator import DataManipulator

import pandas as pd
import argparse

#=============================
# ONE_HOT_CODE
#
#	- Contains methods for processing File I/O && one_hot_coding
#=============================
def main():
	parser = argparse.ArgumentParser(description='One-hot-code data')
	parser.add_argument('input_path', type=str, help='full path to input file')
	parser.add_argument('output_path', type=str, help='full path to output file')
	parser.add_argument('number_max_bins', type=int, help='number of max bins for output data')
	args = parser.parse_args()
	input_path = args.input_path
	output_file = args.output_path
	num_bins = args.number_max_bins

	data_frame = FileManager.get_csv_file_data_pandas(input_path)
	print('data frame head:')
	print(data_frame.head(3))

	hot_coded_data_frame = DataManipulator.one_hot_code(data_frame, num_bins)
	print('data one hot coded head:')
	print(hot_coded_data_frame.head(3))

	hot_coded_data_frame.to_csv(output_file, header=None, index=None, sep=',', mode='a')


'''
if __name__ == '__main__':
	main()
	'''
