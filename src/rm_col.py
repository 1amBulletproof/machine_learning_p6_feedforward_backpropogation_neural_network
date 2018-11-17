#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Script to remove the final column of csv

from file_manager import FileManager
import csv
import argparse
import numpy as np

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to run tests')

	parser = argparse.ArgumentParser(description='Remove the final column')
	parser.add_argument('file_path_in', type=str, help='full path to input file')
	parser.add_argument('file_path_out', type=str, help='full path to output file')
	parser.add_argument('columns', nargs='+', type=int, help='the columns to remove')
	args = parser.parse_args()
	columns = args.columns
	columns = sorted(columns, reverse=True)
	print('deleting these columns in this order')
	print(columns)

	data = FileManager.get_csv_file_data_numpy(args.file_path_in, ',')
	for column in columns:
		data = np.delete(data, column, axis=1)
	data_as_numbers = data.astype(np.float)

	np.savetxt(args.file_path_out, data_as_numbers, delimiter=',')

	'''
	#INPUTS
	print()
	print('INPUTS')
	input_path = args.file_path_in
	print('input file path:', input_path)
	output_path = args.file_path_out
	print('output file path:', output_path)

	#STRIP GIVEN COLUMN
	col_idx = args.column
	with open(input_path, "r") as file_in:
		with open(output_path, "w") as file_out:
			writer = csv.writer(file_out)
			for row in csv.reader(file_in):
				new_row = row[0:col_idx]
				new_row.append(row[col_idx+1:])
				writer.writerow(new_row)
	'''


'''
if __name__ == '__main__':
	main()
'''
