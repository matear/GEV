# script to run the gev code to compute the desire output
#   time python gev1.py 0 >& l0 | tee l0 &
   time python gev1.py 0 >& l0 &
   time python gev1.py 10 >& l10 &
   time python gev1.py 15 >& l15 &
   time python gev1.py 20 >& l20 &
   time python gev1.py 25 >! l25 &

time python ifit.py >& lfit1  &
