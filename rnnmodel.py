import torch
import torch.nn as nn
from torch import optim


class RNN(nn.Module):
	"""
	-- RNN model class --
	Architecture:
	Input layer --> Hidden recurrent layer --> Linear Readout
	"""

	def __init__(self, input_dim, hidden_dim, output_dim, use_cuda):
		super(RNN, self).__init__()
		
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.use_cuda = use_cuda
		
		self.input_to_hidden = nn.Linear(input_dim + hidden_dim, hidden_dim, bias = True)
		self.hidden_to_output = nn.Linear(hidden_dim, output_dim, bias = True)
		self.activation_function = nn.LeakyReLU(0.01)
		
	
	def forward(self, inputs):
		
		N_batches, T = inputs.shape[0], inputs.shape[1]+1
		
		# Initialize the activity of the recurrent layers
		output_activity, hidden_activity = torch.rand(N_batches, self.output_dim), torch.rand(N_batches, self.hidden_dim)

		output_activity *= self.output_dim**0.5
		hidden_activity *= self.hidden_dim**0.5
		
		if self.use_cuda and torch.cuda.is_available():
			output_activity = output_activity.cuda()
			hidden_activity = hidden_activity.cuda()
		
		x, h = [], []

		x.append(output_activity)
		h.append(hidden_activity)

		for t in range(1,T):
			
			combined_inputs_hiddenlayer = torch.cat((inputs[:,t-1,:], hidden_activity),1)
			hidden_activity = self.activation_function(self.input_to_hidden(combined_inputs_hiddenlayer))
			
			output_activity = self.hidden_to_output(hidden_activity)
			
			x.append(output_activity)
			h.append(hidden_activity)

		x = torch.stack(x).transpose(0,1)
		h = torch.stack(h).transpose(0,1)

		return x, h