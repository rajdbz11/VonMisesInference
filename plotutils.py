import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable


def DisplayGraphicalModel(J, node_positions,dpi):
	J_max = np.max(J)
	Ns = J.shape[0]
	fig, ax = plt.subplots(1,2,figsize=(10,3),dpi=dpi)
	for i in range(Ns):
	    for j in range(i+1):
	        if i != j:
	            ax[0].plot([node_positions[i][0], node_positions[j][0]],[node_positions[i][1], node_positions[j][1]],color = 'gray', linewidth = 3*J[i,j,3]/J_max)
	    
	for k in range(Ns):
	    ax[0].plot(node_positions[k][0],node_positions[k][1],'ko',markersize=10)
	ax[0].axis('off')
	ax[0].set(title = 'graphical model')

	pcm = ax[1].imshow(J[...,3])
	ax[1].set(title = 'coupling strengths')
	fig.colorbar(pcm,ax=ax[1])

	plt.show()


def DisplayBeliefDynamics(var, b, local_inputs, beliefs):
	bmax = np.round(100*beliefs[b,var].max())/100
	lmax = np.round(100*local_inputs[b,var].max())/100
	vmax = max(bmax, lmax)

	fig, ax = plt.subplots(1,3,figsize=(15,3),dpi=100,gridspec_kw={'width_ratios': [1,1,1]})

	pcm = ax[0].imshow(local_inputs[b,var],cmap='coolwarm',vmin=0,vmax=vmax)
	ax[0].set_title(r'local potential $\phi_t(s = \theta_i)$',fontsize=14)
	ax[0].set_xlabel(r'time $t$',fontsize=14)
	ax[0].set_ylabel(r'bin $i$',fontsize=14)
	divider = make_axes_locatable(ax[0])
	cax = divider.append_axes("right", size="5%", pad=0.1)
	fig.colorbar(pcm,cax=cax)

	pcm = ax[1].imshow(beliefs[b,var],cmap='coolwarm',vmin=0,vmax=vmax)
	ax[1].set_title(r'belief $b_t(s=\theta_i)$ ',fontsize=14)
	ax[1].set_xlabel(r'time $t$',fontsize=14)
	ax[1].set_ylabel(r'bin $i$',fontsize=14)
	divider = make_axes_locatable(ax[1])
	cax = divider.append_axes("right", size="5%", pad=0.1)
	fig.colorbar(pcm,cax=cax)

	ax[2].plot(local_inputs.flatten(),beliefs[...,1:].flatten(),'b.',markersize=1,alpha=0.4)
	ax[2].plot([0,lmax],[0,bmax],alpha=0.5,color='black')
	ax[2].grid(True)
	ax[2].set_xlabel('discrete feedforward input',fontsize=14)
	ax[2].set_ylabel('discrete beliefs',fontsize=14)

	plt.show()


def ComputeCircularMoments(z, K):
	# z is tensor of beliefs of shape B x Ns x K x T
	# K  = no. of discrete bins used to represent belief
	# B  = no. of batches
	# Ns = no. of variables
	# T  = no. of time steps

	bin_centers  = np.pi*(np.arange(0,K)/K + 0.5/K - 1/2)
	cos2x, sin2x = np.cos(2*bin_centers), np.sin(2*bin_centers)
	cos4x, sin4x = np.cos(4*bin_centers), np.sin(4*bin_centers)

	z_cos2x = np.sum(z*cos2x[None,None,:,None],axis=2)
	z_sin2x = np.sum(z*sin2x[None,None,:,None],axis=2)
	z_cos4x = np.sum(z*cos4x[None,None,:,None],axis=2)
	z_sin4x = np.sum(z*sin4x[None,None,:,None],axis=2)

	return z_cos2x, z_sin2x, z_cos4x, z_sin4x


def DisplayCircularMoments1(local_inputs, beliefs, K):
	l_cos2x, l_sin2x, l_cos4x, l_sin4x = ComputeCircularMoments(local_inputs, K)
	b_cos2x, b_sin2x, b_cos4x, b_sin4x = ComputeCircularMoments(beliefs, K)	
	fig, ax = plt.subplots(1,4,figsize=(16,3),dpi=100)
	ax[0].plot(l_cos2x.flatten(), b_cos2x[...,1:].flatten(),'b.',markersize=1)
	ax[0].set_xlabel('local potential moments', fontsize=14)
	ax[0].set_ylabel('belief moments', fontsize=14)
	ax[0].set_title(r'$c_1(b)$ vs. $c_1(\phi)$', fontsize=14)
	ax[1].plot(l_sin2x.flatten(), b_sin2x[...,1:].flatten(),'b.',markersize=1)
	ax[1].set_title(r'$s_1(b)$ vs. $s_1(\phi)$', fontsize=14)
	ax[2].plot(l_cos4x.flatten(), b_cos4x[...,1:].flatten(),'b.',markersize=1)
	ax[2].set_title(r'$c_2(b)$ vs. $c_2(\phi)$', fontsize=14)
	ax[3].plot(l_sin4x.flatten(), b_sin4x[...,1:].flatten(),'b.',markersize=1)
	ax[3].set_title(r'$s_2(b)$ vs. $s_2(\phi)$', fontsize=14)
	for k in range(4):
	    ax[k].grid(True)
	    #ax[k].axis('square')

	plt.show()


def DisplayCircularMoments2(local_inputs, beliefs, K):
	l_cos2x, l_sin2x, l_cos4x, l_sin4x = ComputeCircularMoments(local_inputs, K)
	b_cos2x, b_sin2x, b_cos4x, b_sin4x = ComputeCircularMoments(beliefs, K)	
	fig, ax = plt.subplots(1,2,figsize=(8,3),dpi=100)
	ax[0].plot(b_cos2x.flatten(), b_sin2x.flatten(),'r.',markersize=1,alpha=0.4)
	ax[0].plot(l_cos2x.flatten(), l_sin2x.flatten(),'b.',markersize=1,alpha=0.2)
	ax[0].set_xlabel(r'$c_1$',fontsize=14)
	ax[0].set_ylabel(r'$s_1$',fontsize=14)
	ax[0].set_title(r'red: $b$, blue: $\phi$')
	ax[0].axis('square')

	ax[1].plot(b_cos4x.flatten(), b_sin4x.flatten(),'r.',markersize=1,alpha=0.4)
	ax[1].plot(l_cos4x.flatten(), l_sin4x.flatten(),'b.',markersize=1,alpha=0.2)
	ax[1].set_xlabel(r'$c_2$',fontsize=14)
	ax[1].set_ylabel(r'$s_2$',fontsize=14)
	ax[1].set_title(r'red: $b$, blue: $\phi$')
	ax[1].axis('square')

	plt.show()

