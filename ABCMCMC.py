"""
This PYTHON script implements a ABC MCMC algorithim to approximate the 
posterior of the succession problem of Bayes and Laplace using a uniform prior 
and a truncated normal normal for the proposal density.

This follows the algorith in 'ABC for dynamical systems' T. Toni et al 2008
"""

import numpy as np 
import scipy.stats as ss 
import matplotlib.pyplot as plt 


def q_sample(theta):
	"""return a sample from the truncated normal centered about theta
						FINISHED!!!
	"""
	r = 2
	while r >= 1 or r <= 0:
		r = np.random.normal(loc=theta, scale=0.1)
	return r

def q(theta_prop,theta_current):
	"""returns the value of the proposal density at theta_prop conditioned on 
	the current postion theta_current
						IN PROGRESS
	"""
	return None

def f(theta):
	"""returns a data set generated with the parameter theta
						IN PROGRESS
	"""
	return None

def d(X_true,X_star):
	"""returns the distance between samples via a metric on a summary statistic
						IN PROGRESS
	"""
	return None

def prior(theta):
	"""returns the value of the prior distribution evaluated at theta
						FINISHED!!!
	"""
	if theta <=1 and theta >= 0:
		y = 1
	else:
		y = 0
	return y


def ABC_MCMC(epsilon, N, X_true):
	"""Carries out a MCMC of N steps and returns the states visited
						FINISHED!!!
	"""
	#-------------------------- M1 --------------------------#
	i = 0
	theta = 0.5
	theta_chain = [theta]
	#-------------------------- M2 --------------------------#
	while i < N:
		theta_star = q_sample(theta)
	    #---------------------- M3 --------------------------#
	    X_star = f(theta_star)
	    #---------------------- M4 --------------------------#
	    if d(X_true,X_star) <= epsilon:
	    	#------------------ M5 --------------------------#
	    	foo = ( prior(theta_star)*q(theta,theta_star) )/\
	    		( prior(theta)*q(theta_star,theta) )
	    	alpha = min(1, foo)
	    	r = np.random.uniform()
	    	if r <= alpha:
	    		theta = theta_star
	    theta_chain += [theta]
	    #---------------------- M6 --------------------------#
	    i += 1
	return theta_chain

def main():
	return None

if __name__ == '__main__':
	main()