#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			11/14/2018
#@description	run experiment

from neural_network_ff_bp import NeuralNetworkFFBP

import argparse
from file_manager import FileManager
#from data_manipulator import DataManipulator
import numpy as np
import pandas as pd

#=============================
# run_model()
#
#	- read-in 5 groups of input data, train on 4/5,
#		test on 5th, cycle the 4/5 & repeat 5 times
#		Record overall result!
#=============================
def run_models_with_cross_validation(number_of_layers, nodes_per_layer, learning_rate, max_epoch, error_thresh):

	#GET DATA
	#- expect data_0 ... data_4
	data_groups = list()
	data_type = 'int'
	data_groups.append(FileManager.get_csv_file_data_array('data_0', data_type))
	data_groups.append(FileManager.get_csv_file_data_array('data_1', data_type))
	data_groups.append(FileManager.get_csv_file_data_array('data_2', data_type))
	data_groups.append(FileManager.get_csv_file_data_array('data_3', data_type))
	data_groups.append(FileManager.get_csv_file_data_array('data_4', data_type))
	
	model1_culminating_result = 0;
	model1_final_average_result = 0

	NUM_GROUPS = len(data_groups)
	#For each data_group, train on all others and test on me
	for test_group_id in range(NUM_GROUPS):
		print()
		#Form training data as 4/5 data
		train_data = list()
		for train_group_id in range(len(data_groups)):
			if (train_group_id != test_group_id):
				#Initialize train_data if necessary
				if (len(train_data) == 0):
					train_data = data_groups[train_group_id]
				else:
					train_data = train_data + data_groups[train_group_id]

		print('train_data group', str(test_group_id), 'length: ', len(train_data))
		#print(train_data)

		test_data = data_groups[test_group_id]
		test_data = pd.DataFrame(test_data)

		train_data = pd.DataFrame(train_data)
		print(train_data.head())
		model1 = NeuralNetworkFFBP(train_data, number_of_layers, nodes_per_layer)
		model1.train(learning_rate, max_epoch, error_thresh)

		print_classifications = False
		if (test_group_id == 0): #Required to print classifications for one fold
			print_classifications = True
		model1_result = model1.test(test_data, print_classifications) # returns (attempts, fails, success)
		model1_accuracy = (model1_result[0]/model1_result[1]) * 100
		print('Accuracy:', model1_accuracy, '%')
		model1_culminating_result = model1_culminating_result + model1_accuracy

	model1_final_average_result = model1_culminating_result / NUM_GROUPS
	#print()
	#print('final average result:')
	#print(final_average_result)
	#print()

	return (model1_final_average_result)


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Run the classification tree test')
	parser.add_argument('num_layers', type=int, help='number of hidden layers in neural network processing data')
	parser.add_argument('num_nodes_per_hidden_layer', type=int, help='number of hidden layers in neural network processing data')
	parser.add_argument('training_rate', type=float, help='rate at which to train')
	parser.add_argument('max_epochs', type=int, help='number of training epochs')
	parser.add_argument('error_threshold', type=float, help='maximum error allowed for stopping point')
	args = parser.parse_args()
	print(args)
	num_nodes_per_hidden_layer = args.num_nodes_per_hidden_layer
	num_layers = args.num_layers
	training_rate = args.training_rate
	max_epochs = args.max_epochs
	error_thresh = args.error_threshold

	final_result = run_models_with_cross_validation(num_layers, num_nodes_per_hidden_layer, training_rate, max_epochs, error_thresh)
	print()
	print('Feedforward Neural Network AVG Accuracy (%):', final_result, '%') 



if __name__ == '__main__':
	main()
