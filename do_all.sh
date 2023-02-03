#!/bin/bash
# scripts to explore the hyperparameters and solutions of the neural network 
source ~/start_miniforge.sh 
conda activate tf_xarray

# solve many times best hyperPs
./best.sh a 64 64 64 1 >& logao
./best.sh b 64 64 64 1 >& logbo

exit

# explore hyperPs for problem a
./model.sh >& loga 

# explore hyperPs problem b
sed -e 's/version="a"/version="b"/g' model.sh > junk.sh
chmod u+x junk.sh 
./junk.sh >& logb


exit
# useful code
# 
seed_value= 0
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)
