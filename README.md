# GEV
Generalised Extreme Value Distribution

analysis to produce information on the number of samples required to
accurately compute a return period.

Trying to write a paper on the topic.

gev1.py   iterative solves for a GEV for different sample sizes

gev_plot.py makes some generic plots

fit.py - my playing with least squares fit to the data but failed

fit2.py - command callable DL script to fit the data
fit3.py - command callable DL script but using fit_lib.py library

fit2a.py - file using fit_lib.py but for producing figures for paper

fit2b.py - as fit2a.py but solving for uncertainty of return value given 
	GEV parameters, number of samples and ARI
fit3b.py - command callable DL script for solving fit2b interactively

ifit.py - iterative script to cycle through multiple DL setups 
model.sh - iterative script to cycle through multiple DL setups 
and fit the data

fit_lib.py - module for fitting tools for the GEV problem




