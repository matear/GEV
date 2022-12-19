#!/bin/bash
# script to explore the hyperparameters of the neural network 
source ~/start_miniforge.sh 
conda activate tf_xarray

layer="0 4 8 16 32 64"
layer0="4 8 16 32 64"
version='b'


for it in 0 1  # for reg=0 and 0.001
do 
echo $it $reg
printf -v reg "%.3f\n" "$(( $it ))e-3"
echo $it $reg
rm his.txt


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
    echo 'unknowns=' $(( 5*$i+$i*($l+l)+$l*($m+1) + $m +1 ))
    time python fit3${version}.py $reg 0.004 400 $i $l $m 
    done
  done
done
mv his.txt ${version}his$it.txt

done

