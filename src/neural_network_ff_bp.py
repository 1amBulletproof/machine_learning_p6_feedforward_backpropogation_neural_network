#!/bin/python
# #@author		Brandon Tarney
#@date			11/10/2018
#@description	Feedforward Neural Networks w/ Backpropogation

from base_model3 import BaseModel
import numpy as np
import pandas as pd


#=============================
# NeuralNetworkFFBP
#
# - Class to encapsulate a Feedforward Neural Network trained w/ Backpropogation
# - *ASSUMPTION*: Classes 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
#=============================
class NeuralNetworkFFBP(BaseModel):

	def __init__(self, train_data, number_of_layers, number_of_hidden_layer_nodes):
		BaseModel.__init__(self, train_data)
		if number_of_layers < 0:
			print('neural network must have 1+ layers, exiting')
			return
		self.data = train_data
		self.number_of_layers = number_of_layers
		self.number_of_hidden_layer_nodes = number_of_hidden_layer_nodes
		self.layers = self.build_neural_network(number_of_layers, number_of_hidden_layer_nodes)

	def build_neural_network(self, number_of_layers, number_of_hidden_layer_nodes):
		#print('build_neural_network:')

		#TODO: Add a bias to every hidden layer

		layers = [] #list of layers (as maps)...layers can contain weights, outputs, deltas
		#Get the actual classes
		number_of_classes = len(self.data[self.data.columns[-1]].unique())
		number_of_input_features = self.data.shape[1] - 1 #no count last col: it is class value
		for layer_idx in range (0, number_of_layers):
			#Layer object as a dictionary
			layer_dictionary = dict()

			#Define the shape of the layer based on which layer it is
			if layer_idx == (number_of_layers - 1):
				#Output layer
				values_shape = (number_of_classes,1)
				if layer_idx == 0:
					#1 layer neural net: inputs feeding output layer
					nodes_shape = (number_of_classes, number_of_input_features)
				else:
					#2+ layer neural net: hidden layers feeding output layer
					nodes_shape = (number_of_classes, number_of_hidden_layer_nodes)
			else:
				values_shape = (number_of_hidden_layer_nodes,1)
				#Hidden layer
				if layer_idx == 0:
					#Use inputs as size
					nodes_shape = (number_of_hidden_layer_nodes, number_of_input_features)
				else:
					#Use previous layer size as input
					nodes_shape = (number_of_hidden_layer_nodes, number_of_hidden_layer_nodes)

			#Using the proper shape, create default weights (-0.01 - 0.01), outputs (0), and deltas (0)
			weights = np.random.randint(-10, 11, nodes_shape) / 1000 #Should create 2d random matrix w/ weights -0.01 - 0.01
			layer_dictionary['weights'] = weights

			deltas = np.zeros(values_shape)
			layer_dictionary['deltas'] = deltas
			outputs = np.zeros(values_shape)
			layer_dictionary['outputs'] = outputs
			errors = np.zeros(values_shape)
			layer_dictionary['errors'] = outputs

			#Add this layer
			layers.append(layer_dictionary)

			#print('created layer:', layer_idx)
			#print(layer_dictionary)

		print('Initial Neural Network:')
		print(layers)
		return layers

	#=============================
	# train()
	#
	#	- train neural network until convergence or max epoch
	#	- effectively determine class weights
	#	- *ASSUMPTION*: CLASSES ARE 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
	#
	#@param	learning_factor	determines how quickly the model learns
	#@param	max_epochs	determine the maximum number of epochs to train-on
	#@return	weight_vectors
	#=============================
	def train(self, learning_rate=0.5, max_epochs=9999):

		#Convert dataframe to numpa
		data_as_np = self.data.values

		#Stopping criteria ('convergence')
		epoch_counter = 0
		total_mean_squared_error = 9999
		error_threshold = 0.1
	
		while (epoch_counter < max_epochs and total_mean_squared_error > error_threshold):
			#print('epoch:', epoch_counter)

			#randomize the inputs every time
			np.random.shuffle(data_as_np)

			#Iterate all inputs, update weights each itr or after all
			for input_idx in range(0, len(data_as_np)):
				#FEEDFORWARD
				#====================
				#Calculate the output at each layer, front to back

				#TODO: don't forget to account for the bias

				#Separate features from target value (class)
				inputs = data_as_np[input_idx]
				target_value = inputs[-1]
				input_no_label = inputs[:-1]
				print('inputs:',inputs, 'target-value', target_value)

				#Calculate the outputs
				next_inputs = input_no_label 
				for layer in self.layers:
					dot_product = np.dot(next_inputs, layer['weights'].T) #Transpose weights to apply inputs per node
					result = self.sigmoid(dot_product)
					result_2d = np.reshape(result, (len(result),1))
					layer['outputs'] = result_2d
					next_inputs = result #next layer feeds the inputs, note they must be 1d array for proper dot product

				#BACKPROPOGATE
				#====================
				#Calculate & save the error/delta at each layer
				for layer_idx in reversed(range(0, self.number_of_layers)):
					#print('BP: layer ', layer_idx)
					layer = self.layers[layer_idx]
					layer_error = layer['errors'] #placeholder

					#ERROR
					if layer_idx == (self.number_of_layers - 1):
						#OUTPUT LAYER
						layer_error = self.get_output_error(target_value, layer)
					else:
						#HIDDEN LAYER
						layer_error = self.get_hidden_layer_error(self.layers[layer_idx+1])

					layer['errors'] = layer_error

					#DELTA
					layer_delta = self.get_delta(layer)
					layer['deltas'] = layer_delta

				#UPDATE WEIGHTS
				#====================
				#Update each layer with weight deltas
				#NOTE can do this every iteration (small learn rate) 
				#		or every epoch (large learning rate)
				next_inputs = input_no_label
				for layer in self.layers:
					print('deltas')
					print(layer['deltas'])
					update_amt = learning_rate * layer['deltas']
					print('update_amt')
					print(update_amt)
					print('update_amt per input')
					print(update_amt * next_inputs)
					layer['weights'] += learning_rate * layer['deltas'] * next_inputs #Transpose deltas to apply each per node, not per weight
					next_inputs = layer['outputs'].flatten()

				print('layers after training:')
				print(self.layers)

			#Update stopping criteria
			epoch_counter += 1
			previous_total_mean_squared_error = total_mean_squared_error
			total_mean_squared_error = self.get_total_mean_squared_error(self.layers[-1]) #Total error of output layer
			#print('total error:', total_mean_squared_error)

		print('Training complete!')
		print('Training done. epoch:', epoch_counter, ', total error:', total_mean_squared_error)
		#print(self.layers)
		return self.layers

	#=============================
	# get_output_error()
	#
	#	- get the error at the output layer (i.e. target - output)
	#	- *ASSUMPTION*: CLASSES ARE 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
	#
	#@param	target_value		effectively the "class", 0-based
	#@param	output_layer		the output layer of the neural network
	#@return	output_error	error shaped to match layer structure
	#=============================
	def get_output_error(self, target_value, output_layer):
		#Setup target_value as vector can be applied to all output neurons
		#Each output neuron represents a class, indexed from 0
		shape_of_layer = output_layer['outputs'].shape
		target_value_as_matrix = np.zeros(shape_of_layer)
		print('target_value_as_matrix')
		target_value_as_matrix[target_value] = 1 
		print(target_value_as_matrix)
		print('outputs')
		print(output_layer['outputs'])
		output_error = target_value_as_matrix - output_layer['outputs']
		print('output error')
		print(output_error)
		#Reshape the output to match the neural network shape, i.e. 1 val per row
		#output_error = np.reshape(output_error, (len(output_error), -1))
		return output_error

	#=============================
	# get_total_mean_squared_error()
	#
	#	- get the total mean squared error 
	#	- *ASSUMPTION*: CLASSES ARE 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
	#
	#@param	layer		layer to calculate total error
	#@return	total_mean_squared_error	
	#=============================
	def get_total_mean_squared_error(self, layer):
		errors = layer['errors']
		total_mean_squared_error = 0.5 * np.sum(np.power(errors,2))
		return total_mean_squared_error

	#=============================
	# get_hidden_layer_error()
	#
	#	- get the error for a hidden layer
	#	- *ASSUMPTION*: next_layer deltas & weights already set
	#	- *ASSUMTPION*: classes 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
	#
	#@param	next_layer	the next layer, used to calculate this layer's error
	#@return	hidden layer error	
	#=============================
	def get_hidden_layer_error(self, next_layer):
		hidden_layer_error = np.sum(np.multiply(next_layer['deltas'], next_layer['weights']))
		return hidden_layer_error

	#=============================
	# get_delta()
	#
	#	- get the delta for any layer
	#	- *ASSUMPTION*: layer error and output already recorderd
	#	- *ASSUMPTION*: CLASSES ARE 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
	#
	#@param	next_layer	the next layer, used to calculate this layer's error
	#@return	hidden layer error	
	#=============================
	#output*(1-output) * error
	def get_delta(self, layer):
		print('getting delta, heres the layer before')
		print(self.layers)
		print('layer[outputs]')
		print(layer['outputs'])
		print('1-layer[outputs]')
		print(1-layer['outputs'])
		print('layer[errors]')
		print(layer['errors'])

		delta = layer['outputs'] * (1 - layer['outputs']) * layer['errors']
		print('delta')
		print(delta)
		return delta

	#=============================
	# sigmoid() #
	#
	#	- calculate the sigmoid fcn
	#
	#@param input_val	input vector
	#@return		vector of sigmoid values
	#=============================
	def sigmoid(self, input_val):
			return 1 / (1 + np.exp(-input_val))
			
	#=============================
	# test() #
	#
	#	- test the internal neural network
	#	- Assumes the model already trained!
	#
	#@param test_data	data to test as dataframe
	#@param print_classifications	boolean to decide whether to display classificaiton
	#@return		classification_accuracy as percentage
	#=============================
	def test(self, test_data, print_classifications=False):
		#Setup the test
		data_as_np = test_data.values
		#print('test data:', data_as_np)
		number_of_tests = len(data_as_np)
		number_prediction_correct = 0

		#Iterate all inputs, find the output & get the classification
		for input_idx in range(0, len(data_as_np)):
				#FEEDFORWARD
				#====================
				#Calculate the output at each layer, front to back, getting the outputs

				#TODO: don't forget to account for the bias

				#Separate features from target value (class)
				inputs = data_as_np[input_idx]
				target_value = inputs[-1]
				input_no_label = inputs[:-1]

				#Calculate the outputs
				layer_output = self.layers[0]['outputs'] #placeholder
				next_inputs = input_no_label 
				for layer in self.layers:
					dot_product = np.dot(next_inputs, layer['weights'].T) #Transpose weights to apply inputs per node
					layer_output = self.sigmoid(dot_product)
					next_inputs = layer_output #next layer feeds the inputs, note they must be 1d array for proper dot product

				print('neural_net output', layer_output)
				predicted_class = np.argmax(layer_output, axis=0)
				if predicted_class == target_value:
					number_prediction_correct += 1

				if print_classifications:
					print('target:', target_value, 'prediction:', predicted_class)

		return (number_prediction_correct, number_of_tests)


def main():

	#Setup the data
	data1 = [[0, 1, 0], [ 1, 0, 1]]
	data1 = pd.DataFrame(data1)
	data2 = [[1,0,0,0], [0,1,0,1], [0,0,1,2]]
	data2 = pd.DataFrame(data2)
	xor_data = [[0, 0, 0], [0, 1, 1], [ 1, 0, 1], [1, 1, 0]] 
	xor_data = pd.DataFrame(xor_data)

	'''
	print()
	print('===========================================')
	print('TEST 1: train the model with no hidden layer')

	data = data2
	print('data')
	print(data)

	number_of_layers = 1
	nodes_per_layer = len(data) - 1 #same number as features
	print('-CREATE neural network:', number_of_layers, 'layers, ', nodes_per_layer, 'nodes')
	neural_net = NeuralNetworkFFBP(data, number_of_layers, nodes_per_layer)

	print()
	learning_rate = 0.07
	max_epoch = 1000
	print('-TRAIN')
	neural_net.train(learning_rate, max_epoch)

	print()
	print('-PREDICT')
	result = neural_net.test(data, True)
	print('accuracy:', float(result[0]/result[1]) * 100, '%')
	print('===========================================')
	print()
	'''
	print()
	print('===========================================')
	print('TEST 2: train the model with 1 hidden layer but linearly separable')
	data = data2
	print('data')
	print(data)

	number_of_layers = 2
	nodes_per_layer = 4
	print('-CREATE neural network:', number_of_layers, 'layers, ', nodes_per_layer, 'nodes')
	neural_net = NeuralNetworkFFBP(data, number_of_layers, nodes_per_layer)

	print()
	learning_rate = 0.05
	max_epoch = 1
	print('-TRAIN')
	neural_net.train(learning_rate, max_epoch)

	print()
	print('-PREDICT')
	neural_net.test(data, True)
	print('===========================================')
	print()
'''

    # ----------- XOR Function -----------------
	print()
	print('TEST 3: train the model with 1 hidden layer, non-linearly separable')
	data = xor_data
	print('data')
	print(data)

	print()
	number_of_layers = 1
	nodes_per_layer = len(data) - 1 #same number as features
	print('neural network:', number_of_layers, 'layers, ', nodes_per_layer, 'nodes')
	neural_net = NeuralNetworkFFBP(data, number_of_layers, nodes_per_layer)

	print()
	learning_rate = 0.07
	max_epoch = 1000
	print('learn the model (learning rate:', learning_rate, ')')
	neural_net.train(learning_rate, max_epoch)

	print()
	print('test the model')
	neural_net.test(data, True)

	print()
	print('PREDICT 4: train the model with 2 hidden layer, non-linearly separable')
	data = xor_data
	print('data')
	print(data)

	print()
	number_of_layers = 1
	nodes_per_layer = len(data) - 1 #same number as features
	print('neural network:', number_of_layers, 'layers, ', nodes_per_layer, 'nodes')
	neural_net = NeuralNetworkFFBP(data, number_of_layers, nodes_per_layer)

	print()
	learning_rate = 0.07
	max_epoch = 1000
	print('learn the model (learning rate:', learning_rate, ')')
	neural_net.train(learning_rate, max_epoch)

	print()
	print('test the model')
	neural_net.test(data, True)
	'''
	
if __name__ == "__main__":
	main()
