#!/bin/bash
# script to explore the hyperparameters of the neural network 
source ~/start_miniforge.sh 
conda activate tf_xarray
rm his.txt

layer="0 4 8 16 32 64"
layer0="4 8 16 32 64"
reg=0.001

for i in $layer0
do 
  for l in $layer0
  do
    for m in $layer
    do 
    layer1=$i
    layer2=$l
    layer3=$m
    echo $layer1 $layer2 $layer3 $reg
    python fit3.py $reg 0.004 400 $i $l $m 
#    echo par=5*$i+$i*($l+l)+$l*($m+1) + $m +1
    done
  done
done


