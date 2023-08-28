import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import time
from pandas import DataFrame

import copy
#import device_use
#Choose device
#device = device_use.device_use

#Rework device managment:
#Model should be fully moved to any desired device by calling model.to(device)
#This means any tensors floating arround and manually moved to device should be covered by model.to(device)
#I.e. taking the device of paramteter or being registered as buffer or simpl overwriting the to(device) method


# MLP
#Basis for NF coupling layer, to represent RQS parameters later

class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_layers=4, neg_slope=0.2):
        super().__init__()
        ins = torch.ones(n_layers)*n_hidden
        ins[0] = n_in
        outs = torch.ones(n_layers)*n_hidden
        outs[-1] = n_out
        
        
        Lin_layers = list(map(nn.Linear, ins.type(torch.int), outs.type(torch.int)))
        ReLU_layers = [nn.LeakyReLU(neg_slope) for _ in range(n_layers)]
        self.network = nn.Sequential(*itertools.chain(*zip(Lin_layers,ReLU_layers)))
        
    def forward(self, x):
        return self.network(x)
    
#Invertible 1x1 convolution like in GLOW paper
#Generalise Permutation with learnt W matrix
class GLOW_conv(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        W_initialize = nn.init.orthogonal_(torch.randn(self.n_dim, self.n_dim))
        #P, L_, U_ = torch.lu_unpack(*torch.linalg.lu_factor(W_initialize))
        #P, L_, U_ = torch.lu_unpack(*torch.lu(W_initialize))
        P, L_, U_ = torch.linalg.lu(W_initialize)
        #P = P.to(device)
        #L_ = L_.to(device)
        #U_ = U_.to(device)
        
        
        #P not changed but it needs to be stored in the state_dict
        self.register_buffer("P", P)
        
        #Declare as model parameters
        #Diagonal of U sourced out to S
        S_ = torch.diagonal(U_)
        self.S = nn.Parameter(S_)
        self.L = nn.Parameter(L_)
        #Declare with diagonal 0s, without changing U_ and thus S_
        self.U = nn.Parameter(torch.triu(U_, diagonal=1))
        
    def _get_W_and_ld(self):
        #Make sure the pieces stay in correct shape as in GLOW
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.n_dim).to(self.L.device))
        U = torch.triu(self.U, diagonal=1)
        S = torch.diag(self.S)
        
        W = self.P@L@(U+S)
        logdetW = torch.sum(torch.log(torch.abs(self.S)))
        ##Debug
        #print("W:",W)
        #print("P:",self.P)
        #print("L:",L)
        #print("U:",U)
        #print("S:",S)
        return W, logdetW
    
    #Pass condition as extra argument, that is not used in the convolution
    #it stayes untouched, does not get permuted with values that
    #will be transformed
    def forward(self, x, x_condition):
        W, logdetW = self._get_W_and_ld()
        ##Debug
        #print(torch.mean(torch.abs(x)))
        y = x@W
        #print(torch.mean(torch.abs(y)))
        #print("W:",W)
        #print("NEW")

        ## Debug
        #logdetW = torch.sum(torch.log(torch.abs(self.S)))
        return y, logdetW
    
    def backward(self, y, x_condition):
        W, logdetW_inv = self._get_W_and_ld()
        #Just a minus needed
        logdetW_inv = -logdetW_inv
        W_inv = torch.linalg.inv(W)
        x = y@W_inv
        
        ##Debug
        #logdetW_inv = -torch.sum(torch.log(torch.abs(self.S)))
        
        return x, logdetW_inv

#To evaluate a spline we need to know in which bin
#x falls.
def find_bin(values, bin_boarders):
    #Make shure that a value=uppermost boarder is in last bin not last+1
    bin_boarders[..., -1] += 10**-6
    return torch.sum((values.unsqueeze(-1)>=bin_boarders),dim=-1)-1

#Write a function that takes a parametrisation of RQS and points and evaluates RQS or inverse
#Splines from [-B,B] identity else
#Bin widths normalized for 2B interval size
def eval_RQS(X, RQS_bin_widths, RQS_bin_heights, RQS_knot_derivs, RQS_B, inverse=False):
    #Get boarders of bins as cummulative sum as in NSF paper
    #As they are normalized this goes up to 2B
    #Represents upper boarders
    bin_boardersx = torch.cumsum(RQS_bin_widths, dim=-1)
    #We shift so that they cover actual interval [-B,B]
    bin_boardersx -= RQS_B
    #Now make sure we include all boarders i.e. include lower boarder B
    #Also for bin determination make sure upper boarder is actually B and
    #doesn't suffer from rounding (order 10**-7, searching algorithm would anyways catch this)
    bin_boardersx = F.pad(bin_boardersx, (1,0), value=-RQS_B)
    bin_boardersx[...,-1] = RQS_B
    
    #update widths?


    #Same for heights
    bin_boardersy = torch.cumsum(RQS_bin_heights, dim=-1)
    bin_boardersy -= RQS_B
    bin_boardersy = F.pad(bin_boardersy, (1,0), value=-RQS_B)
    bin_boardersy[...,-1] = RQS_B
    
    
    #Now with completed parametrisation (knots+derivatives) we can evaluate the splines
    #For this find bin positions first
    
    ##Debug
    #print(X.shape)
    #print(bin_boardersx.shape)
    
    bin_nr = find_bin(X, bin_boardersy if inverse else bin_boardersx)
    
    #After we know bin number we need to get corresponding knot points and derivatives for each X
    
    #Debug
    #print(bin_boardersx.shape, bin_nr.unsqueeze(-1).shape)
    #print(bin_nr)
    
    x_knot_k = bin_boardersx.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
    x_knot_kplus1 = bin_boardersx.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    
    y_knot_k = bin_boardersy.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
    y_knot_kplus1 = bin_boardersy.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    
    delta_knot_k = RQS_knot_derivs.gather(-1, bin_nr.unsqueeze(-1)).squeeze(-1)
    delta_knot_kplus1 = RQS_knot_derivs.gather(-1, (bin_nr+1).unsqueeze(-1)).squeeze(-1)
    
    s_knot_k = (y_knot_kplus1-y_knot_k)/(x_knot_kplus1-x_knot_k)
    
    if inverse:
        a = (y_knot_kplus1-y_knot_k)*(s_knot_k-delta_knot_k)+(X-y_knot_k)*(delta_knot_kplus1+delta_knot_k-2*s_knot_k)
        b = (y_knot_kplus1-y_knot_k)*delta_knot_k-(X-y_knot_k)*(delta_knot_kplus1+delta_knot_k-2*s_knot_k)
        c = -s_knot_k*(X-y_knot_k)
        
        Xi = 2*c/(-b-torch.sqrt(b**2-4*a*c))
        Y = Xi*(x_knot_kplus1-x_knot_k)+x_knot_k
        
        dY_dX = (s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))**2
        #No sum yet, so we can later keep track which X weren't in the intervall and need logdet 0
        logdet = -torch.log(dY_dX)

        ##Debug Variant

        #logdet = -torch.log(s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2)) + 2*torch.log(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))
    else:
        ##Debug
        Xi = (X-x_knot_k)/(x_knot_kplus1-x_knot_k)
        #print("Xi",Xi.shape)
        Y = y_knot_k+((y_knot_kplus1-y_knot_k)*(s_knot_k*Xi**2+delta_knot_k*Xi*(1-Xi)))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))
        #print("Y",Y.shape)
        dY_dX = (s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2))/(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))**2
        #print("dY",dY_dX)
        logdet = torch.log(dY_dX)



        ## Debug variant
        #logdet = torch.log(s_knot_k**2*(delta_knot_kplus1*Xi**2+2*s_knot_k*Xi*(1-Xi)+delta_knot_k*(1-Xi)**2)) - 2*torch.log(s_knot_k+(delta_knot_kplus1+delta_knot_k-2*s_knot_k)*Xi*(1-Xi))
        
    return Y, logdet

def RQS_global(X, RQS_bin_widths, RQS_bin_heights, RQS_knot_derivs, RQS_B, inverse=False):
    inside_interval = (X<=RQS_B) & (X>=-RQS_B)
    Y = torch.zeros_like(X)
    logdet = torch.zeros_like(X)
    
    ##Debug
    #print("split_int",X[inside_interval],X[~inside_interval])
    #print(X.shape)
    #print(inside_interval.shape,RQS_bin_widths.shape)
    Y[inside_interval], logdet[inside_interval] = eval_RQS(X[inside_interval], RQS_bin_widths[inside_interval,:], RQS_bin_heights[inside_interval,:], RQS_knot_derivs[inside_interval,:], RQS_B, inverse)
    
    Y[~inside_interval] = X[~inside_interval]
    logdet[~inside_interval] = 0
    
    #Now sum the logdet, zeros will be in the right places where e.g. all X components were 0
    logdet = torch.sum(logdet, dim=1)
    
    return Y, logdet


def one_fn(*args):
    return 1


class NSF_CL(nn.Module):
    def __init__(self, dim_notcond, dim_cond, split=0.5, K=8, B=3, network = MLP, network_args=(16,4,0.2)):
        super().__init__()
        self.dim = dim_notcond
        self.dim_cond = dim_cond
        self.K = K
        self.B = B
        
        self.split1 = int(self.dim*split)
        self.split2 = self.dim-self.split1
        
        self.net = network(self.split1, (3*self.K-1)*self.split2, *network_args)
        
        #Decide if conditioned or not
        if self.dim_cond>0:
            self.net_cond = network(dim_cond, (3*self.K-1)*self.split2, *network_args)
            
        
    def forward(self, x, x_cond):
        ##Debug
        #print("beginning fw NSF_CL",x)
        unchanged, transform = x[..., :self.split1], x[..., self.split1:]
        #print("after split",unchanged,transform)
        #Conditioned or not
        if self.dim_cond>0:
            thetas = (self.net_cond(x_cond)*self.net(unchanged)).reshape(-1, self.split2, 3*self.K-1)
        else:
            thetas = self.net(unchanged).reshape(-1, self.split2, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
        
        transform, logdet = RQS_global(transform, widths, heights, derivs, self.B)
        ##Debug
        #print("after rqs",unchanged,transform)
        return torch.hstack((unchanged,transform)), logdet
    
    def backward(self, x, x_cond):
        unchanged, transform = x[..., :self.split1], x[..., self.split1:]
        
        if self.dim_cond>0:
            thetas = (self.net_cond(x_cond)*self.net(unchanged)).reshape(-1, self.split2, 3*self.K-1)
        else:
            thetas = self.net(unchanged).reshape(-1, self.split2, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
        
        transform, logdet = RQS_global(transform, widths, heights, derivs, self.B, inverse=True)
        
        return torch.hstack((unchanged,transform)), logdet
    
    
class NSF_CL2(nn.Module):
    def __init__(self, dim_notcond, dim_cond, K=8, B=3, network = MLP, network_args=(16,4,0.2)):
        super().__init__()
        self.dim = dim_notcond
        self.dim_cond = dim_cond
        self.K = K
        self.B = B
        
        self.split = self.dim//2
        
        #Works only for even?
        self.net1 = network(self.split, (3*self.K-1)*self.split, *network_args)
        self.net2 = network(self.split, (3*self.K-1)*self.split, *network_args)
        
        if dim_cond>0:
            self.net_cond1 = network(dim_cond, (3*self.K-1)*self.split, *network_args)
            self.net_cond2 = network(dim_cond, (3*self.K-1)*self.split, *network_args)
        
    def forward(self, x, x_cond):
        first, second = x[..., self.split:], x[..., :self.split]
        
        if self.dim_cond>0:
            thetas = (self.net_cond1(x_cond)*self.net1(second)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net1(second).reshape(-1, self.split, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
            
        first, logdet = RQS_global(first, widths, heights, derivs, self.B)
        
        if self.dim_cond>0:
            thetas = (self.net_cond2(x_cond)*self.net2(first)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net2(first).reshape(-1, self.split, 3*self.K-1)
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
        
        second, logdet_temp = RQS_global(second, widths, heights, derivs, self.B)
            
        logdet += logdet_temp
            
        return torch.hstack((second,first)), logdet
        
    def backward(self, x, x_cond):
        first, second = x[..., self.split:], x[..., :self.split]
        
        if self.dim_cond>0:
            thetas = (self.net_cond2(x_cond)*self.net2(first)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net2(first).reshape(-1, self.split, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
            
        second, logdet = RQS_global(second, widths, heights, derivs, self.B, inverse=True)
        
        if self.dim_cond>0:
            thetas = (self.net_cond1(x_cond)*self.net1(second)).reshape(-1, self.split, 3*self.K-1)
        else:
            thetas = self.net1(second).reshape(-1, self.split, 3*self.K-1)
        
        widths, heights, derivs = torch.split(thetas, self.K, dim=-1)
        widths = F.softmax(widths, dim=-1)*2*self.B
        heights = F.softmax(heights, dim=-1)*2*self.B
        derivs = F.softplus(derivs)
        derivs = F.pad(derivs, pad=(1,1), value=1)
            
        first, logdet_temp = RQS_global(first, widths, heights, derivs, self.B, inverse=True)
            
        logdet += logdet_temp
            
        return torch.hstack((second,first)), logdet
        
# ---- The following functions are needed for NF-sails sampling ----




# ---- End of functions needed for NF-sails sampling ----

class NSFlow(nn.Module):
    def __init__(self, n_layers, dim_notcond, dim_cond, CL, **kwargs_CL):
        super().__init__()
        self.dim = dim_notcond
        self.dim_cond = dim_cond


        '''coupling_layers = itertools.repeat(CL(dim_notcond, dim_cond, **kwargs_CL), n_layers)
        conv_layers = itertools.repeat(GLOW_conv(dim_notcond), n_layers)'''
        coupling_layers = [CL(dim_notcond, dim_cond, **kwargs_CL) for _ in range(n_layers)]
        conv_layers = [GLOW_conv(dim_notcond) for _ in range(n_layers)]


        self.layers = nn.ModuleList(itertools.chain(*zip(conv_layers,coupling_layers)))
        
        self.prior = torch.distributions.Normal(torch.zeros(dim_notcond), torch.ones(dim_notcond))

        #Information about hyperparameters accessible from outside
        #The function _get_right_hypers below will then reconstruct this back to __init__ arguments, if you change the model, change both.
        #This is needed in the background for recreating the same model in some cases like sampling with multiprocessing
        kwargs_parsed = kwargs_CL.copy()
        kwargs_parsed["network"] = "MLP"
        self.give_kwargs = {"n_layers":n_layers, "dim_notcond":dim_notcond, "dim_cond":dim_cond, "CL":"CL2"if CL==NSF_CL2 else "CL", **kwargs_parsed}
        
    def forward(self, x, x_cond):
        logdet = torch.zeros(x.shape[0]).to(x.device)
        
        for layer in self.layers:
            ##Debug
            #print(x)
            x, logdet_temp = layer.forward(x, x_cond)
            logdet += logdet_temp
            
        #Get p_z(f(x)) which is needed for loss function together with logdet
        prior_z_logprob = self.prior.log_prob(x).sum(-1)
        
        return x, logdet, prior_z_logprob
    
    def backward(self, x, x_cond):
        logdet = torch.zeros(x.shape[0]).to(x.device)
        
        for layer in reversed(self.layers):
            x, logdet_temp = layer.backward(x, x_cond)
            logdet += logdet_temp
            
        return x, logdet
    
    def sample_Flow(self, number, x_cond):
        return self.backward(self.prior.sample(torch.Size((number,))), x_cond)[0]
    
    #NF sails sampling (Sampling with Langevin dynamics in the latent space) both local and global exploration
    #Local is RMMALA, global is independent MH

    def sample_sails(self, number, x_cond, chain_length, time_step, probability):
        if number != x_cond.shape[0]:
            raise ValueError("Number of samples and number of conditions must be the same")
        #We it would be optimal to have only one markov chain, but:
        #1. We need different markov chains for each condition as the pdf is different
        #2. We can expect a large incease in speed if we split the sampling into different markov chains (while sacrificing some accuracy)

        #Determine the number of chains needed
        #Number of unique conditions in x_cond
        unique_cond, n_unique_cond = torch.unique(x_cond, dim=0, return_counts=True)
        #Number of chains needed for each condition if chain length is chain_length
        n_chains = torch.ceil(n_unique_cond/chain_length).int()
        n_chains_total = n_chains.sum()
        #How many are over the desired amount, throw them away later
        n_too_many = n_chains*chain_length - n_unique_cond

        #Now prepare the conditions for sampling, one for each chain
        x_cond_sample = torch.repeat_interleave(unique_cond, n_chains, dim=0)

        #Prepare the samples
        samples = torch.zeros(chain_length+1, n_chains_total, self.dim).to(x_cond.device)

        #Draw first sample for each chain
        z_k = self.sample_Flow(n_chains_total, x_cond_sample)
        samples[0] = z_k

        for k in range(chain_length):
            z_k = samples[k]
            #Choose kernel by uniform probability
            u = torch.rand(n_chains_total)
            #Local or global exploration
            z_k_plus_1 = torch.where(u<probability, self._sample_RMMALA(z_k, x_cond_sample, time_step).T, self.sample_I_MH(z_k, x_cond_sample).T).T
            #Store samples
            samples[k+1] = z_k_plus_1

        #For each condition, throw away the extra samples
        #(not implemented yet....) #Maybe in loop throw away last markov samples from last chain (i.e. set nan)
        #Get the chain indices with cumsum n_chains
        #Append condition (x_cond_sample) to samples (for sorting)
        samples = torch.cat((samples, x_cond_sample.repeat(chain_length+1,1,1)), dim=-1)
        #Stack together while ignoring the first sample
        samples = samples[1:].reshape(-1, self.dim+self.dim_cond)

        #Restore the original order (sort by condition)
        #Get map from sorted to original
        arg_order_x_cond = torch.argsort(torch.argsort(x_cond, dim=0)[:,0], dim=0)
        #Sort samples by condition
        argsort_samples = torch.argsort(samples[:,-self.dim_cond:], dim=0)[:,0]
        samples = samples[argsort_samples]
        #Sort back to original order (apply map)
        samples = samples[arg_order_x_cond]

        samples = samples[:, :-self.dim_cond]
        return samples
    
    #RMMALA sampling
    def _sample_RMMALA(self, z_k, x_cond, time_step):
        #Draw candidate
        xi = self.prior.sample(z_k[0].shape)
        #Compute score
        score = self._latent_score(z_k, x_cond)
        z_prime = 1


    def _latent_score(self, z, x_cond):
        #(s^tilde_Z(z))
        #s^tilde_Z(z) = J^-1(z)*grad_z log q^tilde_Z(z) = grad_x log q_X(x)
        #i.e. the latent score is equal to the score of q_X(x) in the original space
        x = self.backward(z, x_cond)[0]
        with torch.inference_mode(False):
            x = x.clone()
            x.requires_grad_(True)
            #To ignore model parameter gradients and only track x gradients:
            for param in self.parameters():
                param.requires_grad_(False)
            _, logdet, prior_z_logprob = self.forward(x, x_cond)#Here model gradients are still tracked, which is not needed
            log_prob = logdet + prior_z_logprob
            score = torch.autograd.grad(log_prob, x, grad_outputs=torch.ones_like(log_prob))[0]
            score = score.detach()
            for param in self.parameters():
                param.requires_grad_(True)
        return score
    
    def jacobian_matrix(self, z, x_cond, mode="forward"):
        #Compute Jacobian of forward or backward transformation
        #Just use automatic differentiation. Parts of J coud be tracked analytically, but this is not implemented
        #Since autograd is anyways used for The other parts and we ptoentially have a lot of layers i.e. a lot of matrix multiplications, this should not be too slow (i hope)
        
        #First turn of gradient tracking for model parameters
        for param in self.parameters():
            param.requires_grad_(False)

        #Turn off inference mode
        with torch.inference_mode(False):
            True

    
    def to(self, device):
        super().to(device)
        self.prior = torch.distributions.Normal(torch.zeros(self.dim).to(device), torch.ones(self.dim).to(device))
        return self


#Function to get the right hyperparameters for the model.give_kwargs update this if you change the model
def _get_right_hypers(model_params):
    kwargs_dict = copy.deepcopy(model_params)

    kwargs_dict["CL"] = NSF_CL2 if kwargs_dict["CL"] == "CL2" else NSF_CL
    kwargs_dict["network"] = MLP if kwargs_dict["network"] == "MLP" else MLP

    return kwargs_dict


def train_flow(flow_obj:NSFlow, data:DataFrame, cond_names:list, epochs, lr=2*10**-2, batch_size=1024, loss_saver=None, checkpoint_dir=None, gamma=0.998, give_textfile_info=False, optimizer_obj=None):
    """
    Train a normalizing flow model on the given data.

    Parameters
    ----------

    flow_obj : NSFlow instance
        The flow model to train.
    data : pd.DataFrame
        The data to train on. IMPORTANT: Although colums are usually safely regarded to by name, here the order of the colums is important:
        The order of 'data.columns' will be the order of any samples the model generates, it will not use the names of the columns.
        Thus make sure to remember it, see example below.
    cond_names : list of str
        The names of the conditional variables.
    epochs : int
        The number of epochs to train.
    lr : float (optional), default : 2*10**-2
        The initial learning rate.
    batch_size : int (optional), default : 1024
        The batch size to use.
    loss_saver : list (optional), default : None
        A list to store the loss values in. Values are appended live during training.
    checkpoint_dir : str (optional), default : None
        The directory to save checkpoints in. If None, no checkpoints are saved. E.g. "saves/checkpoints/".
    gamma : float (optional), default : 0.998
        The learning rate decay factor of the exponential learning rate scheduler. Decreases the learning rate by a factor of gamma every 10 batches,
        or every 120 batches once the learning rate is below 3*10**-6.
    give_textfile_info : str (optional), default : False
        If not False, the given string is used as the suffix of a textfile in which information about the training is saved.
    optimizer_obj : torch.optim.Optimizer instance (optional), default : None
        The optimizer to use. If None, the RAdam optimizer is used. E.g. torch.optim.Adam(flow_obj.parameters(), lr=lr).

    Returns
    -------
    None

    Example
    -------
    >>> components = ["x", "y", "z", "M_tot", "average_age"]
    >>> conditions = ["M_tot", "average_age"]
    >>> data = np.random.normal(size=(1000000, len(components)))
    >>> data = pd.DataFrame(data, columns=components)
    >>> flow_obj = NSFlow(8, dim_notcond=3, dim_cond=2, ...)
    >>> losses = []
    >>> train_flow(flow_obj, data, conditions, epochs=100, loss_saver=losses)
    >>> sample = flow_obj.sample_Flow(1000, torch.from_numpy(data[conditions].values).type(torch.float)).numpy()#will crate a numpy array of shape (1000, 3):
    >>> sample
    array(...)
    >>> #Columns are in the order of the components, but unnamed.
    >>> sample = pd.DataFrame(sample, columns=components)
    >>> sample
    x         y         z
    ...
    
    Note that the orer already matterd when passing the conditions to sample_Flow.
    """
    
    #Device the model is on
    device = flow_obj.parameters().__next__().device

    #Infos to printout
    n_steps = data.shape[0]*epochs//batch_size+1

    #Get index based masks for conditional variables
    mask_cond = np.isin(data.columns.to_list(), cond_names)
    mask_cond = torch.from_numpy(mask_cond).to(device)
    #Convert DataFrame to tensor (index based)
    data = torch.from_numpy(data.values).type(torch.float)
    
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    if optimizer_obj is None:
        optimizer = optim.RAdam(flow_obj.parameters(), lr=lr)
    else:
        optimizer = optimizer_obj
    
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    

    if not give_textfile_info==False:
        with open(f"status_output_training_{give_textfile_info}.txt", mode="w+") as f:
            f.write("")
    

    #Safe losses
    losses = []

    #Total number of steps
    ct = 0
    #Total number of checkpoints
    cp = 0

    start_time = time.perf_counter()

    for e in range(epochs):
        for i, batch in enumerate(data_loader):
            
            x = batch.to(device)
            
            #Evaluate model
            z, logdet, prior_z_logprob = flow_obj(x[...,~mask_cond],x[...,mask_cond])
            
            #Get loss
            loss = -torch.mean(logdet+prior_z_logprob)
            losses.append(loss.item())
            
            #Set gradients to zero
            optimizer.zero_grad()
            #Compute gradients
            loss.backward()
            #Update parameters
            optimizer.step()
            
            #Every 100 steps print out some info to console or file and save to loss history if specified
            if ct % 100 == 0:
                if not give_textfile_info==False:
                    with open(f"status_output_training_{give_textfile_info}.txt", mode="a") as f:
                        f.write(f"Step {ct} of {n_steps}, Loss:{np.mean(losses[-50:])}, lr={lr_schedule.get_last_lr()[0]}\n")
                else:
                    print(f"Step {ct} of {n_steps}, Loss:{np.mean(losses[-50:])}, lr={lr_schedule.get_last_lr()[0]}")
                if loss_saver is not None:
                    loss_saver.append(np.mean(losses[-50:]))
            
            #Every 5000 steps save model and loss history (checkpoint)
            if checkpoint_dir != None and ct % 5000 == 0 and not ct == 0:
                torch.save(flow_obj.state_dict(), f"{checkpoint_dir}checkpoint_{cp%2}.pth")
                curr_time = time.perf_counter()
                np.save(f"{checkpoint_dir}losses_{cp%2}.npy", np.array(loss_saver+[curr_time-start_time]))
                cp += 1
            
            ct += 1

            #Decrease learning rate every 10 steps until it is smaller than 3*10**-6, then every 120 steps
            if lr_schedule.get_last_lr()[0] <= 3*10**-6:
                decrease_step = 120
            else:
                decrease_step = 10

            #Update learning rate every decrease_step steps
            if ct % decrease_step == 0:
                lr_schedule.step()
