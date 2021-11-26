# script to run the gev code to compute the desire output
#   time python gev1.py 10 >& 1 | tee l10 &
   time python gev1.py 10 >& l10 &
   time python gev1.py 15 >& l15 &
   time python gev1.py 20 >& l20 &
   time python gev1.py 25 >! l25 &
