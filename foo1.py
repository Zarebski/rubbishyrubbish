import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)

def f(x):
	v = np.random.normal() # process noise
	x_new = x + v		   # system model
	return x_new

def h(x):
	theta = 3 				 			# measurement parameter
	w = np.sqrt(0.5)*np.random.normal()	# measurement noise
	z = theta*x + w 					# measurement model
	return z

def Xpath(t):
	dW = np.random.normal(size=t)
	W = np.cumsum(dW)
	W = np.hstack(([0],W))
	return W

def Zpath(x_path):
	t = len(x_path) - 1
	Z = np.zeros(t)
	for i in range(t):
		Z[i] = h(x_path[i+1])
	return Z

def seed_particles(N):
	m1 = np.zeros(2*N)
	m2 = np.hstack((5*np.ones(N),3*np.ones(N)))
	m3 = 0.5*np.ones(2*N)/N
	m = np.vstack((m1,m2,m3))
	return m

def move_particles(m,z):
	num_particles = len(np.transpose(m))
	z_vect = z*np.ones(num_particles)
	ZZZ = np.random.normal(size=num_particles)
	old_pos = m[0,:]
	theta = m[1,:]
	new_pos = 0.5*( old_pos + ZZZ +  z_vect/theta)
	m_new = m
	m_new[0,:] = new_pos
	return m_new 

# def update_weights(m, z):
# 	num_particles = len(np.transpose(m))
# 	z_vect = z*np.ones(num_particles)
# 	w_old = m[2,:]
# 	x = m[0,:]
# 	theta = m[1,:]
# 	w_new = w_old*np.exp( -( ( theta*x - z )**2 ) )
# 	w_new /= np.sum(w_new)
# 	m_new = m
# 	m_new[2,:] = w_new
# 	return m_new

def resample_particles(m):
	num_particles = len(np.transpose(m))
	a = range(num_particles)
	c = np.random.choice(a,size=num_particles,p=m[2,:])
	m_resamp = m[:,c]
	m_resamp = m_resamp[[0,1],:]
	w_unif = np.ones(num_particles)/num_particles
	m_resamp = np.vstack((m_resamp,w_unif))
	return m_resamp



def main():
	t = 10
	XX = Xpath(t)
	ZZ = Zpath(XX)
	TT = range(t+1)
	MM = np.zeros(t+1)

	m = seed_particles(10**6)
	print np.sum( m[1,:]*m[2,:] )

	for i in range(t):
		m = move_particles(m)
		m = update_weights(m, ZZ[i])
		m = resample_particles(m)
		MM[i+1] = np.sum( m[0,:]*m[2,:] )
		print np.sum( m[1,:]*m[2,:] )


	plt.figure()
	plt.plot(TT,XX,'b')
	plt.plot(TT[1:],ZZ,'r--')
	plt.plot(TT,MM,'g')
	plt.show()

	

if __name__ == '__main__':
	main()
