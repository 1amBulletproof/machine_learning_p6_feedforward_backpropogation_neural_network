#!/bin/sh

echo "Script to vary the number of layers in a neural network"
echo "Specify the number of nodes per hidden layer, learning rate, and max_error"

if [ $1 == "-h" ]; then
	echo "Specify the number of nodes per hidden layer, learning rate, and max_error"
	exit 0
fi

node_num=$1
learning_rate=$2
max_epoch=1000
max_error=$3

max_layers=4
layer_num=1
increment=1

while [ $layer_num != $max_layers ]
do
	echo "layers: $layer_num"
	filename="nn_"$layer_num"_"$node_num"_"$learning_rate"_"$max_epoch"_"$max_error
	echo "filename: $filename"
	#echo "test" > $filename
	python3 run_neural_network.py $layer_num $node_num $learning_rate $max_epoch $max_error > $filename
	layer_num=$(($layer_num + $increment))
done

