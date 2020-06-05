import numpy as np
from numpy import pi
from scipy import special
from scipy import signal
from sklearn.datasets import make_sparse_spd_matrix
import matplotlib.pyplot as plt
import time
import scipy.io
from tqdm import tqdm

def VonMises(x,k,m):
	"""
	Unnormalized von mises function
	x : input; domain is [0,pi)
	k : concentration parameters
	m : preferred orientation
	"""
	return np.exp(k*np.cos(2*(x - m)))


def VonMisesCoupling(x1,x2,m1,m2,J):
	"""
	Unnormalized von mises pairwise potential 
	x1, x2 : inputs; domain for each is [0,pi)
	m1, m2 : preferred orientations of the two variables
	J      : array of 4 coupling terms cos-cos, cos-sin, sin-cos, sin-sin
	"""
	c1, c2 = np.cos(2*(x1-m1)), np.cos(2*(x2-m2))
	s1, s2 = np.sin(2*(x1-m1)), np.sin(2*(x2-m2))
	Jcc, Jcs, Jsc, Jss = J[0], J[1], J[2], J[3]
	return np.exp( Jcc*c1*c2 + Jcs*c1*s2 + Jsc*s1*c2 + Jss*s1*s2 )


def VonMises2D(x1,x2,k1,k2,m1,m2,J):
	"""
	Unnormalized bivariate von mises density
	x1, x2 : inputs; domain for each is [0,pi)
	k1, k2 : concentrations of the two variables
	m1, m2 : preferred orientations of the two variables
	J      : array of 4 coupling terms cos-cos, cos-sin, sin-cos, sin-sin
	"""
	return VonMises(x1,k1,m1)*VonMises(x2,k2,m2)*VonMisesCoupling(x1,x2,m1,m2,J)


def DiscreteSingletonPotentials(N, KVec, MuVec):
	"""
	Function to discretize von mises singleton potentials
	
	N     : no. of discrete bins
	KVec  : vector of concentration parameters
	MuVec : vector of preferred orienations
	
	"""

	x =  np.arange(0,N)/N + 0.5/N - 1/2
	x *= pi

	if len(MuVec.shape) == 1:
		x = x[None,:]
		KVec, MuVec = KVec[:,None], MuVec[:,None]
	else:
		MuVec 	= MuVec[:,None,:]			# MuVec has shape Ns x T, 	make it  Ns x 1 x T
		KVec 	= KVec[:,None,None]			# KVec has shape Ns, 		make it  Ns x 1 x 1
		x 		= x[None,:,None]			# x has shape N, 			make it   1 x N x 1

	return VonMises(x, KVec, MuVec)*pi/N


def DiscretePairwisePotentials(N, MuVec, J):
	"""
	Function to discretize von mises pairwise potentials
	
	N     : no. of discrete bins
	MuVec : vector of preferred orienations of size Ns (static) or Ns x T (dynamic case)
	J     : coupling interactions tensor of size Ns x Ns x 4
	
	"""
	
	# define grid for evaluating the discrete pairwise potentials
	x =  np.arange(0,N)/N + 0.5/N - 1/2
	x *= pi
	x1, x2 = np.meshgrid(x, x)

	# adjacency matrix
	AdjMat = (np.sum(np.abs(J),axis=2) != 0)*1 

	# initialize the discrete potentials
	Ns = MuVec.shape[0]

	if len(MuVec.shape) == 1:
		Psi 	= np.ones([Ns,Ns,N,N])
		dynamic = False
	else:
		T 		= MuVec.shape[1]
		Psi 	= np.ones([Ns,Ns,N,N,T])
		dynamic = True

	
	for i in range(Ns):
		for j in range(i):
			if AdjMat[i,j]:
				if dynamic:
					Psi[i,j] = VonMisesCoupling(x1[:,:,None], x2[:,:,None], MuVec[None,None,i,:], MuVec[None,None,j,:], J[i,j])
					Psi[j,i] = Psi[i,j].transpose(1,0,2)

				else:
					Psi[i,j] = VonMisesCoupling(x1,x2,MuVec[i],MuVec[j],J[i,j])
					Psi[j,i] = Psi[i,j].T
				
	Psi = Psi*((pi/N)**2)
	
	return Psi


def DiscreteBP(Phi, Psi, AdjMat, MaxIters, lam, eps):
	"""
	Function that performs BP on a discrete multivariate distribution of Ns variables
	
	Inputs:
	
	Phi      : Singleton potentials. static case : Ns x N, 			dynamic : Ns x N x T
	Psi      : Pairwise potentials.	 static case : Ns x Ns x N x N, dynamic : Ns x Ns x N x N x T
	AdjMat   : Adjacency matrix of the graphical model
	MaxIters : maximum no. of iterations to run in static case
	lam      : update constant for messages
	eps      : threshold for convergence
	
	Outputs:
	messages_t : all messages pairs as a function of time, size Ns x Ns x N x T
	beliefs_t  : normalized marginals as a function of time, size Ns x N xT 
	err_t      : MSE between old and new messages as a function of time 
	"""
	
	# Define message[i,j] as the message from node i to node j
	
	Ns, N = Phi.shape[0], Phi.shape[1] # No. of variables, no. of discrete bins for each variable

	if len(Phi.shape) == 3:
		dynamic  	= True
		MaxIters 	= Phi.shape[2]
	else:
		dynamic 	= False

	

	# Initialize messages: each message is a vector of length N
	messages_old, messages_new, messages_t = np.ones([Ns,Ns,N]), np.ones([Ns,Ns,N]), np.ones([Ns,Ns,N,MaxIters])
	
	# Initialize beliefs
	beliefs_t = np.ones([Ns,N,MaxIters+1])
	if dynamic:
		beliefs_t[:,:,0] = Phi[:,:,0]/np.sum(Phi[:,:,0,None],axis=1)
	else:
		beliefs_t[:,:,0] = Phi/np.sum(Phi[:,:,None],axis=1)
	
	err_t = np.zeros(MaxIters)

	for t in range(MaxIters):
		for i in range(Ns):
			for j in range(Ns):
				if AdjMat[i,j]:
					
					# multiply all incoming messages to node i, excluding message from j to i
					l = list(range(Ns))
					l.remove(j)
					#incoming = np.prod(messages_old[l,i],axis=0)
					incoming = np.exp(np.sum(np.log(messages_old[l,i]),axis=0))
					
					# integrate incoming messages with local evidence and pairwise potential
					if dynamic:
						integrate = np.expand_dims(Phi[i,:,t],axis=0)*np.expand_dims(incoming,axis=0)*Psi[i,j,:,:,t]
					else:
						integrate = np.expand_dims(Phi[i],axis=0)*np.expand_dims(incoming,axis=0)*Psi[i,j]
					
					# marginalize
					marginal = np.sum(integrate,axis=1)
					
					# normalize
					marginal = marginal/sum(marginal)
					
					# update message
					messages_new[i,j] = (1-lam)*messages_old[i,j] + lam*marginal

		# compute beliefs
		for i in range(Ns):
			if dynamic:
				b = Phi[i,:,t]*np.prod(messages_new[:,i],axis=0)
			else:
				b = Phi[i]*np.prod(messages_new[:,i],axis=0)
			beliefs_t[i,:,t+1] = b/sum(b)

		messages_t[...,t] = messages_new*1.0

		if not dynamic:
			# check for convergence using mean squared error
			if np.sum(AdjMat)>0:
				err = np.sum((messages_new - messages_old)**2)/np.sum(AdjMat)
			else:
				err = 1
				
			err_t[t] = err

			if err < eps:
				print('Converged in', t+1, 'iterations')
				break
		
		# update message old	
		messages_old = messages_new*1.0
		

	if not dynamic:
		if t == MaxIters-1:
			print('Did not converge in', MaxIters, 'iterations.')
		else:
			messages_t, beliefs_t, err_t = messages_t[...,0:t+1], beliefs_t[...,0:t+2], err_t[0:t+1]
	
	
	return messages_t, beliefs_t, err_t



def BPWrapper(Ns, KVec, MuVec, J, K, MaxIters, lam, eps):
	
	AMat = (np.sum(np.abs(J),axis=2) != 0)*1    

	
	Phi = DiscreteSingletonPotentials(K, KVec, MuVec)
	Psi = DiscretePairwisePotentials(K, MuVec, J)

	# 1. Run BP
	t_st = time.time()
	messages_t, beliefs_t, err_t = DiscreteBP(Phi, Psi, AMat, MaxIters, lam, eps)
	t_en = time.time()
	print('Time taken for BP = %.2f s' %(t_en-t_st))


	return Phi, Psi, messages_t, beliefs_t, err_t



def ConditionalVonMises(i, KVec, J, s):
	"""
	Function that returns the parameters of the univariate conditional distribution (which is von Mises),
	p(s_i|rest of the variables), for a multivariate von mises distribution
	
	Inputs
	i     : index of variable for which we compute the conditional distribution
	KVec  : concentration vector of the joint distribution
	J     : tensor of couplings of the joint distribution
	s     : current values of all the variables
	
	Outputs
	kappa : concentration parameter of the conditional distribution
	mu    : mean of the univariate conditional distrubution
	"""
	
	Ns = len(KVec)

	cVec, sVec = np.cos(s), np.sin(s)
	
	A = KVec[i] + np.sum(J[i,:,0]*cVec + J[i,:,1]*sVec)
	B = np.sum(J[i,:,2]*cVec + J[i,:,3]*sVec)
	
	kappa = np.sqrt(A**2 + B**2)
	mu = np.arctan2(B,A)

	return kappa, mu


def GibbsSamplingIteration(KVec, MuVec, J, s):
	"""
	Function that performs one complete iteration of Gibbs sampling for the multivariate von mises distribution.
	One complete iteration involves sampling from all univariated conditional distributions of s_i, i = 1,..., N_s

	Inputs
	KVec  : concentration vector of the joint distribution
	MuVec : mean vector of the joint distribution
	J     : tensor of couplings of the joint distribution
	s     : current sample vector
	
	Outputs
	s     : updated sample vector
	"""
	Ns = len(KVec)
	for i in range(Ns):
		kappa, mu = ConditionalVonMises(i, KVec, J, s)
		s[i] = np.random.vonmises(mu, kappa, 1)
	
	return s


def GibbsSamplerVonMises(KVec, MuVec, J, N_samples, T_burnin, T_skip):
	"""
	Gibbs sampler for joint von mises distribution
	Inputs
	KVec      : concentration vector of the joint distribution
	MuVec     : mean vector of the joint distribution
	J         : tensor of couplings of the joint distribution
	N_samples : no. of output samples needed
	T_burnin  : burn-in period
	T_skip    : no. of samples to skip
	"""
	
	Ns = len(KVec) # no. of variables
	
	sVec = np.random.vonmises(0*MuVec,KVec) #initial value for the joint sample

	# burn-in iterations
	for t in range(T_burnin):
		sVec = GibbsSamplingIteration(KVec, MuVec, J, sVec)

	# start recording samples
	s_samples = np.zeros([Ns,N_samples])

	for k in range(N_samples):
		for t in range(T_skip):
			sVec = GibbsSamplingIteration(KVec, MuVec, J, sVec)
		s_samples[:,k] = sVec

	# Change interval to [-pi/2, pi) and add the means to the samples
	s_samples = s_samples/2
	s_samples = s_samples + np.expand_dims(MuVec,axis=1)
	
	# wrap-around samples outside [-pi/2, pi)
	idx = np.nonzero(s_samples > pi/2)
	s_samples[idx] = s_samples[idx] - pi
	idx = np.nonzero(s_samples < -pi/2)
	s_samples[idx] = pi + s_samples[idx]
	
	return s_samples


def GibbsMarginals(s_samples, bin_edges):
	Ns, K = s_samples.shape[0], len(bin_edges)-1
	marginals = np.zeros([Ns,K])
	for i in range(Ns):
		out = np.histogram(s_samples[i],bin_edges)[0]
		marginals[i] = out/sum(out)
	return marginals


def GibbsSamplingWrapper(KVec, MuVec, J, K, N_samples, T_burnin, T_skip):

	# 2. Gibbs sampling
	t_st = time.time()
	s_samples = GibbsSamplerVonMises(KVec, MuVec, J, N_samples, T_burnin, T_skip)
	t_en = time.time()
	print('Time taken for Gibbs sampling = %.2f s' %(t_en-t_st))

	bin_edges = pi*np.arange(0,K+1)/K - pi/2

	s_marginals = GibbsMarginals(s_samples, bin_edges)


	return s_samples, s_marginals


def CreateGridGraphicalModel(nrows, ncols, c_spatial, c_strength, jitter):

	# creates a grid structured graphical model 
	# interaction strength based on distance between nodes
	# c_spatial - spatial scale for pairwise coupling strength
	# c_strength - pairwise coupling for neighbors at distance 1
	# jitter - standard deviation of jitter in coupling strength

	Ns = nrows*ncols # No. of variables

	# set interaction strength based on distance between the pair of nodes
	node_positions = []
	for row in range(nrows):
	    for col in range(ncols):
	        node_positions.append([row,col])

	J = np.zeros([Ns,Ns,4]) # we will use only the 4th component corresponding to the sine-sine interactions here

	for i in range(Ns):
	    for j in range(i+1):
	        if i != j: 
	            distance = (node_positions[i][0] - node_positions[j][0])**2 + (node_positions[i][1] - node_positions[j][1])**2
	            J[i,j,3] = J[j,i,3] = c_strength*(np.exp(-distance/c_spatial) + jitter*np.random.randn(1))/np.exp(-1/c_spatial)

	# threshold J
	J = (J > 0.05)*J

	return J, node_positions


def PiecewiseConstantNoise(Ny, T, T_const):
	# creates a piece wise constant noise signal
    L       = T//T_const + 1*(T%T_const != 0)
    yInd    = 0.5*np.random.randn(Ny,L)
    yMat 	= []
    # Repeat each independent y for T_const time steps
    for t in range(T):
        yMat.append(yInd[:,t//T_const])

    return np.array(yMat).T


def GenerateDynamicMu(Ns, B, T, T_low, T_high):
	# generate dynamic input orientation signal
	
	#smoothing_filter = signal.hamming(7,sym=True) 
	smoothing_filter = signal.windows.hann(8)
	smoothing_filter = smoothing_filter/sum(smoothing_filter)


	z_real, z_imag = [], []
	for b in range(B):
		T_const = np.random.randint(low=T_low,high=T_high)
		z_real.append(signal.filtfilt(smoothing_filter, 1, PiecewiseConstantNoise(Ns, T, T_const)))
		z_imag.append(signal.filtfilt(smoothing_filter, 1, PiecewiseConstantNoise(Ns, T, T_const)))
	    

	return np.arctan2(np.array(z_imag), np.array(z_real))/2


def generateCouplings(Ns, sp, K):
	"""
	Generate a sparse, random coupling tensor of size Ns x Ns x 4 for an undirected graphical model with Ns random varibles
	Each pairwise potential is characterized by 4 parameters: cos-cos, cos-sin, sin-cos and sin-sin interactions
	
	Ns : No. of variables in the joint distribution
	sp : sparsity, i.e, fraction of zero entries in adjacency matrix
	K  : overall scaling factor of coupling strengths
	"""
	
	# Generate adjacency matrix for undirected graphical model
	AdjMat = (np.tril(np.random.rand(Ns,Ns) > sp,-1))*1.0
	AdjMat = AdjMat + AdjMat.T

	# Generate coupling matrices: Jcc, Jcs, Jsc, Jss
	J = np.zeros([Ns,Ns,4])
	
	Jcc = np.tril(np.random.randn(Ns,Ns),-1)
	Jss = np.tril(np.random.randn(Ns,Ns),-1)
	Jcs = np.tril(np.random.randn(Ns,Ns),-1)
	Jsc = np.tril(np.random.randn(Ns,Ns),-1)
	
	J[:,:,0] = Jcc + Jcc.T
	J[:,:,1] = Jcs + Jsc.T
	J[:,:,2] = Jsc + Jcs.T
	J[:,:,3] = Jss + Jss.T
	
	J = K*J*np.expand_dims(AdjMat,2)
	
	return J