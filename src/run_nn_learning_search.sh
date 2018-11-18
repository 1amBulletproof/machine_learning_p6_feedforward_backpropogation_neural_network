#!/bin/sh

echo "Script to vary the learning rate in a neural network"
echo "Specify the number of layers, number of nodes, and max_error"

if [ $1 == "-h" ]; then
	echo "Specify the number of layers, number of nodes, and max_error"
	exit 0
fi

layer_num=$1
node_num=$2
max_epoch=1000
max_error=$3

max_learning_rate=1.5
learning_rate=0.1
increment=0.1

while [ 1 -eq $(echo "${learning_rate} < ${max_learning_rate}" | bc) ]
do
	echo "learning_rate: $learning_rate"
	filename="nn_"$layer_num"_"$node_num"_"$learning_rate"_"$max_epoch"_"$max_error
	echo "filename: $filename"
	#echo "test" > $filename
	python3 run_neural_network.py $layer_num $node_num $learning_rate $max_epoch $max_error > $filename
	learning_rate=$(echo "scale=2; $learning_rate + $increment" | bc)
done

