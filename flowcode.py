import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import time

#Choose device
device = "cuda:5"


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
        W_initialize = nn.init.orthogonal_(torch.randn(self.n_dim, self.n_dim).to(device))
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
        L = torch.tril(self.L, diagonal=-1)+ torch.diag(torch.ones(self.n_dim).to(device))
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
        
class NSFlow(nn.Module):
    def __init__(self, n_layers, dim_notcond, dim_cond, CL, **kwargs_CL):
        super().__init__()
        self.dim = dim_notcond


        '''coupling_layers = itertools.repeat(CL(dim_notcond, dim_cond, **kwargs_CL), n_layers)
        conv_layers = itertools.repeat(GLOW_conv(dim_notcond), n_layers)'''
        coupling_layers = [CL(dim_notcond, dim_cond, **kwargs_CL) for _ in range(n_layers)]
        conv_layers = [GLOW_conv(dim_notcond) for _ in range(n_layers)]


        self.layers = nn.ModuleList(itertools.chain(*zip(conv_layers,coupling_layers)))
        
        self.prior = torch.distributions.Normal(torch.zeros(dim_notcond).to(device), torch.ones(dim_notcond).to(device))
        
    def forward(self, x, x_cond):
        logdet = torch.zeros(x.shape[0]).to(device)
        
        for layer in self.layers:
            ##Debug
            #print(x)
            x, logdet_temp = layer.forward(x, x_cond)
            logdet += logdet_temp
            
        #Get p_z(f(x)) which is needed for loss function together with logdet
        prior_z_logprob = self.prior.log_prob(x).sum(-1)
        
        return x, logdet, prior_z_logprob
    
    def backward(self, x, x_cond):
        logdet = torch.zeros(x.shape[0]).to(device)
        
        for layer in reversed(self.layers):
            x, logdet_temp = layer.backward(x, x_cond)
            logdet += logdet_temp
            
        return x, logdet
    
    def sample_Flow(self, number, x_cond):
        return self.backward(self.prior.sample(torch.Size((number,))), x_cond)[0]



def train_flow(flow_obj, data, cond_indx, epochs, optimizer_obj=None, lr=2*10**-2, batch_size=1024, loss_saver=None, gamma=0.998, give_textfile_info=False, print_fn=None, **print_fn_kwargs):
    #Infos to printout
    
    n_steps = data.shape[0]*epochs//batch_size+1
    
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    if optimizer_obj is None:
        optimizer = optim.RAdam(flow_obj.parameters(), lr=lr)
    else:
        optimizer = optimizer_obj
    
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    
    start_time = time.perf_counter()
    
    
    #Masks for conditional variable
    mask_cond = torch.full((data.shape[1],), False)
    mask_cond[cond_indx] = True
    mask_cond = mask_cond.to(device)
    #Safe losses
    losses = []

    ct = 0
    for e in range(epochs):
        for i, batch in enumerate(data_loader):
            ##Debug
            #print("new batch")
            x = batch.to(device)
            
            #Evaluate model
            z, logdet, prior_z_logprob = flow_obj(x[...,~mask_cond],x[...,mask_cond])
            
            #Get loss
            loss = -torch.mean(logdet+prior_z_logprob)
            losses.append(loss.item())
            
            #zero_grad() on model (sometimes safer that optimizer.zero_grad())
            flow_obj.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ct % 100 == 0:
                if give_textfile_info:
                    with open("status_output_training.txt", mode="a") as f:
                        f.write(f"Step {ct} of {n_steps}, Loss:{np.mean(losses[-50:])}, lr={lr_schedule.get_last_lr()[0]}\n")
                else:
                    print(f"Step {ct} of {n_steps}, Loss:{np.mean(losses[-50:])}, lr={lr_schedule.get_last_lr()[0]}")
                if loss_saver is not None:
                    loss_saver.append(np.mean(losses[-50:]))
            
            ct+=1
            if ct % 10 == 0:
                lr_schedule.step()
