#!/bin/bash
# script to fit the neural network multiple times with the preferred model 
source ~/start_miniforge.sh 
conda activate tf_xarray

version='a'
# 32 8 32
# 8 16 64
#i=8 l=16 m=64
i=32; l=8; m=32
version=$1
i=$2; l=$3; m=$4; ii=$5

ireg=0
for it in $ii   #  for reg=0 and 0.001
do 
echo $it $reg
printf -v reg "%.3f\n" "$(( $it ))e-3"
printf -v reg "%.3f\n" "$(( $ireg ))e-3"
echo $it $reg
rm his.txt
cp head.txt his.txt


for jtimes in `seq 0 1 50`
do 
    layer1=$i
    layer2=$l
    layer3=$m
    echo $layer1 $layer2 $layer3 $reg $it $jtimes
    time python fit3${version}.py $it $reg 0.004 120 $i $l $m 
    mv model_0 ${version}model${it}_${jtimes}
done
sed -e 's/\[/\ /g' -e 's/\]/\ /g' his.txt > tt.txt
mv tt.txt o${version}his$it.txt

done

