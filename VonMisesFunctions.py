import numpy as np
from numpy import pi
from scipy import special
from sklearn.datasets import make_sparse_spd_matrix
import matplotlib.pyplot as plt
import time
import scipy.io

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
    J      : 2x2 coupling matrix
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
    J      : 2x2 coupling matrix
    """
    return VonMises(x1,k1,m1)*VonMises(x2,k2,m2)*VonMisesCoupling(x1,x2,m1,m2,J)


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

def DiscreteSingletonPotentials(N, KVec, MuVec):
    """
    Function to discretize von mises singleton potentials
    
    N     : no. of discrete bins
    KVec  : vector of concentration parameters
    MuVec : vector of preferred orienations
    
    """

    # x = np.expand_dims(pi*(np.arange(0,N)/N + 0.5/N),axis=0)
    x = np.arange(0,N)/N + 0.5/N - 1/2
    x = np.expand_dims(pi*x,axis=0)
    
    KVec, MuVec = np.expand_dims(KVec,axis=1), np.expand_dims(MuVec,axis=1)
    
    return VonMises(x, KVec, MuVec)*pi/N


def DiscretePairwisePotentials(N, MuVec, J):
    """
    Function to discretize von mises pairwise potentials
    
    N     : no. of discrete bins
    MuVec : vector of preferred orienations
    J     : tensor of coupling parameters
    
    """
    Ns = len(MuVec)
    Psi = np.ones([Ns,Ns,N,N])
    
    AdjMat = (np.sum(np.abs(J),axis=2) != 0)*1 #adjacency matrix

    # x1 ,x2 = np.meshgrid(np.pi*(np.arange(0,N)/N + 0.5/N), np.pi*(np.arange(0,N)/N + 0.5/N))
    
    x = np.arange(0,N)/N + 0.5/N - 1/2
    x1, x2 = np.meshgrid(pi*x, pi*x)
    
    for i in range(Ns):
        for j in range(i):
            if AdjMat[i,j]:
                Psi[i,j] = VonMisesCoupling(x1,x2,MuVec[i],MuVec[j],J[i,j])
                Psi[j,i] = Psi[i,j].T
                
    Psi = Psi*((pi/N)**2)
    
    return Psi

def DiscreteBP(Phi, Psi, AdjMat, MaxIters, lam, eps):
    """
    Function that performs BP on a discrete multivariate distribution of Ns variables
    
    Inputs:
    
    Phi      : Singleton potentials Ns x N
    Psi      : Pairwise potential Ns x Ns x N x N
    AdjMat   : Adjacency matrix of the graphical model
    MaxIters : maximum no. of iterations to run
    lam      : update constant for messages
    eps      : threshold for convergence
    
    Outputs:
    messages_t : all messages pairs as a function of time, size Ns x Ns x N x T
    beliefs_t  : normalized marginals as a function of time, size Ns x N xT 
    err_t      : MSE between old and new messages as a function of time 
    """
    
    # Define message[i,j] as the message from node i to node j
    
    Ns, N = np.shape(Phi) # No. of variables in the distribution, no. of values each random variable takes (no. of discrete bins)

    # Initialize messages: each message is a vector of length N
    #messages_old, messages_new, messages_t = np.ones([Ns,Ns,N])/N, np.ones([Ns,Ns,N]), np.ones([Ns,Ns,N,MaxIters])
    messages_old, messages_new, messages_t = np.ones([Ns,Ns,N]), np.ones([Ns,Ns,N]), np.ones([Ns,Ns,N,MaxIters])
    
    # Initialize beliefs
    beliefs_t = np.ones([Ns,N,MaxIters+1])
    beliefs_t[:,:,0] = Phi/np.expand_dims(np.sum(Phi,axis=1),axis=1)
    
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
                    integrate = np.expand_dims(Phi[i],axis=0)*np.expand_dims(incoming,axis=0)*Psi[i,j]
                    
                    # marginalize
                    marginal = np.sum(integrate,axis=1)
                    
                    # normalize
                    marginal = marginal/sum(marginal)
                    
                    # update message
                    messages_new[i,j] = (1-lam)*messages_old[i,j] + lam*marginal

        # compute beliefs
        for i in range(Ns):
            b = Phi[i]*np.prod(messages_new[:,i],axis=0)
            beliefs_t[i,:,t+1] = b/sum(b)

        messages_t[...,t] = messages_new*1.0

        # check for convergence using mean squared error
        if np.sum(AdjMat)>0:
            err = np.sum((messages_new - messages_old)**2)/np.sum(AdjMat)
        else:
            err = 1
            
        err_t[t] = err

        if err < eps:
            print('Converged in', t+1, 'iterations')
            break
        else:
            messages_old = messages_new*1.0
    
    if t == MaxIters-1:
        print('Did not converge in', MaxIters, 'iterations.')
    else:
        messages_t, beliefs_t, err_t = messages_t[...,0:t+1], beliefs_t[...,0:t+2], err_t[0:t+1]
    
    
    return messages_t, beliefs_t, err_t

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


def BPWrapper(Ns, KVec, MuVec, J, K, MaxIters, lam, eps):
    
    AMat = (np.sum(np.abs(J),axis=2) != 0)*1    

    # 1. Run BP
    Phi = DiscreteSingletonPotentials(K, KVec, MuVec)
    Psi = DiscretePairwisePotentials(K, MuVec, J)

    t_st = time.time()
    messages_t, beliefs_t, err_t = DiscreteBP(Phi, Psi, AMat, MaxIters, lam, eps)
    t_en = time.time()
    print('Time taken for BP = ',np.round(1000*(t_en-t_st))/1000, 's')


    return Phi, Psi, messages_t, beliefs_t, err_t


def GibbsSamplingWrapper(KVec, MuVec, J, K, N_samples, T_burnin, T_skip):

    # 2. Gibbs sampling
    t_st = time.time()
    s_samples = GibbsSamplerVonMises(KVec, MuVec, J, N_samples, T_burnin, T_skip)
    t_en = time.time()
    print('Time taken for Gibbs sampling = ',np.round(1000*(t_en-t_st))/1000, 's')

    bin_edges = pi*np.arange(0,K+1)/K - pi/2

    s_marginals = GibbsMarginals(s_samples, bin_edges)


    return s_samples, s_marginals



# def InferenceWrapper(Ns, s_strength, c_sparsity, c_strength, K, MaxIters, lam, eps, N_samples, T_burnin, T_skip):
    
	
#     # Singleton factor parameters
#     KVec, MuVec = s_strength*np.random.rand(Ns), np.random.rand(Ns)*pi

#     # Pairwise factor parameters
#     J = generateCouplings(Ns, c_sparsity, c_strength)

#     AMat = (np.sum(np.abs(J),axis=2) != 0)*1    

#     # 1. Run BP
#     Phi = DiscreteSingletonPotentials(K, KVec, MuVec)
#     Psi = DiscretePairwisePotentials(K, MuVec, J)

#     t_st = time.time()
#     messages_t, beliefs_t, err_t = DiscreteBP(Phi, Psi, AMat, MaxIters, lam, eps)
#     t_en = time.time()
#     print('Time taken for BP = ',np.round(1000*(t_en-t_st))/1000, 's')

#     # 2. Gibbs sampling
#     t_st = time.time()
#     s_samples = GibbsSamplerVonMises(KVec, MuVec, J, N_samples, T_burnin, T_skip)
#     t_en = time.time()
#     print('Time taken for Gibbs sampling = ',np.round(1000*(t_en-t_st))/1000, 's')

#     bin_edges = pi*np.arange(0,K+1)/K - pi/2

#     s_marginals = GibbsMarginals(s_samples, bin_edges)


#     return AMat, Phi, Psi, messages_t, beliefs_t, err_t, s_samples, s_marginals
