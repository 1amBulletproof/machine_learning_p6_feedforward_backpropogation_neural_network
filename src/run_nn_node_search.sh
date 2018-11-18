#!/bin/sh

echo "Script to vary the number of nodes in the hidden layers of a neural network"
echo "Specify the number of layers, starting number of nodes, learning rate, and max_error"

if [ $1 == "-h" ]; then
	echo "Specify the number of layers, starting number of nodes, learning rate, and max_error"
	exit 0
fi

layer_num=$1
node_num=$2
learning_rate=$3
max_epoch=1000
max_error=$4

max_nodes=20
increment=1

while [ $node_num != $max_nodes ]
do
	echo "nodes: $node_num"
	filename="nn_"$layer_num"_"$node_num"_"$learning_rate"_"$max_epoch"_"$max_error
	echo "filename: $filename"
	#echo "test" > $filename
	python3 run_neural_network.py $layer_num $node_num $learning_rate $max_epoch $max_error > $filename
	node_num=$(($node_num + $increment))
done

