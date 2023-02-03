#!/bin/bash
# script to explore the hyperparameters of the neural network 
source ~/start_miniforge.sh 
conda activate tf_xarray

layer="4 8 16 32 64"
layer0="4 8 16 32 64"
version="a"


ireg=0 # no regularisation 
for it in 0 1  # mse and abs fits
do 
echo $it $reg
printf -v reg "%.3f\n" "$(( $it ))e-3"
printf -v reg "%.3f\n" "$(( $ireg ))e-3"
echo $it $reg
rm his.txt
cp head.txt his.txt


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
    time python fit3${version}.py $it $reg 0.004 120 $i $l $m 
    done
  done
done
sed -e 's/\[/\ /g' -e 's/\]/\ /g' his.txt > tt.txt
mv tt.txt ${version}his$it.txt

done

#sed -e 's/\[/\ /g' -e 's/\]/\ /g' bhis1_mse.txt > tt.txt

# set schedule
from keras.callbacks import LearningRateScheduler

def schedule(epoch):
    if epoch < 10:
        return 0.01
    else:
        return 0.001

lr_scheduler = LearningRateScheduler(schedule)

model.fit(X, y, callbacks=[lr_scheduler])

