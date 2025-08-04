import gc
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from .vsgd import VSGD
from .module import DCL
from .module import VariationalEncoder, MaskedLinear, LoRALinear

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', 
                                       nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)

# #-----------------------------------------------------------------------------
# # Horizontal

class multivib(torch.nn.Module):
    def __init__(self,
                 n_input_a=2000,
                 n_input_b=2000,
                 n_hidden=256, n_latent=10, 
                 n_batch=1,
                 mask=None, relation='negative'):
        super(multivib, self).__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        
        self.maskedlinear = MaskedLinear(self.n_input_b, self.n_input_a)
        if mask is not None:
            self.maskedlinear.set_mask(mask)
        self.translator = torch.nn.Sequential(
            self.maskedlinear,
            torch.nn.BatchNorm1d(self.n_input_a)
        )
        
        self.encoderA = VariationalEncoder(
            n_input=self.n_input_a, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecterA = torch.nn.Linear(self.n_latent+self.n_batch, 128)
        
        self.apply(init_weights)
        
        if relation=='postive':
            initial_weights = torch.ones((self.n_input_a, self.n_input_b))
        else:
            initial_weights = -torch.ones((self.n_input_a, self.n_input_b))
        
        if mask is not None:
            initial_weights[mask!=1]=0
        self.translator[0].weight.data = initial_weights.data.to(self.translator[0].weight.device, 
                                                                 self.translator[0].weight.dtype)

    def forward(self, x_a, x_b, batcha, batchb):
        
        x_BtoA = self.translator(x_b)
        qz_b, z_b = self.encoderA(x_BtoA)
        qz_a, z_a = self.encoderA(x_a)
        
        z_a = self.transform(z_a)
        z_b = self.transform(z_b)
        
        p_a = self.projecterA(torch.cat((z_a, batcha), 1))
        p_b = self.projecterA(torch.cat((z_b, batchb), 1))
            
        return {'z_a': z_a, 'z_b': z_b,
                'qz_a': qz_a, 'qz_b': qz_b,
                'proj_a': p_a, 'proj_b':p_b}

class multivibLoRA(torch.nn.Module):
    def __init__(self,
                 n_input_a=2000,
                 n_input_b=2000,
                 n_hidden=256, 
                 n_latent=10, rank=128,
                 n_batch=1):
        
        super(multivibLoRA, self).__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.rank = rank
        
        self.translator = LoRALinear(self.n_input_b, self.n_input_a, self.rank)
        
        self.encoderA = VariationalEncoder(
            n_input=self.n_input_a,
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecterA = torch.nn.Linear(self.n_latent+self.n_batch, 128)

    def forward(self, x_a, x_b, batcha, batchb):
        
        x_BtoA = self.translator(x_b)
        qz_b, z_b = self.encoderA(x_BtoA)
        qz_a, z_a = self.encoderA(x_a)
        
        z_a = self.transform(z_a)
        z_b = self.transform(z_b)
        
        p_a = self.projecterA(torch.cat((z_a, batcha), 1))
        p_b = self.projecterA(torch.cat((z_b, batchb), 1))
            
        return {'z_a': z_a, 'z_b': z_b,
                'qz_a': qz_a, 'qz_b': qz_b,
                'proj_a': p_a, 'proj_b':p_b}

def multivib_training(model,
                      Xa, Xb,
                      batcha, batchb,
                      epoch=100, batch_size=128, 
                      temp=0.15, alpha=0.05, # beta=0.0,
                      random_seed=0):
    
    contrastive_loss = DCL(temperature=temp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = VSGD(model.parameters(), lr=0.0006, ps=1e-7, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                           factor=0.8,
                                                           patience=20, 
                                                           min_lr=0.0001, 
                                                           verbose=True)
    
    loss_history = []
    for e in range(epoch):
            
        model.to(device)
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xa.shape[0])
        X_train_A = Xa[r,:]
        y_batch_A = batcha[r,:]
        X_tensor_A=torch.tensor(X_train_A).float()
        y_tensor_A=torch.tensor(y_batch_A).float()
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xb.shape[0])
        X_train_B = Xb[r,:]
        y_batch_B = batchb[r,:]
        X_tensor_B=torch.tensor(X_train_B).float()
        y_tensor_B=torch.tensor(y_batch_B).float()
        
        n = min(Xa.shape[0], Xb.shape[0])
        
        total_loss = []
        
        with tqdm(total=n//batch_size, 
                  desc=f"Epoch {e+1}/{epoch}",
                  unit="batch",
                  bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False,
                  position=0) as pbar:
            
            for i in range(n//batch_size):
                
                pbar.update(1)
                
                opt.zero_grad()
                
                inputs_a1 = X_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_a2 = X_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_a = y_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_a1.shape
                inputs_a1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_a2 += torch.normal(0, 1.0, (c, m)).to(device)
            
                inputs_b1 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_b2 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_b = y_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_b1.shape
                inputs_b1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_b2 += torch.normal(0, 1.0, (c, m)).to(device)
                
                out_unpair1 = model(inputs_a1,inputs_b1, batch_a, batch_b)
                out_unpair2 = model(inputs_a2,inputs_b2, batch_a, batch_b)
            
                # Contrastive loss
                cont_loss = (
                    contrastive_loss(out_unpair1['proj_a'], out_unpair2['proj_a'])
                    + contrastive_loss(out_unpair1['proj_b'], out_unpair2['proj_b'])
                )
                
                # KL divergence
                pz = Normal(
                    torch.zeros_like(out_unpair1['qz_a'].mean),
                    torch.ones_like(out_unpair1['qz_a'].mean),
                )
            
                kl_loss = (
                    kl(out_unpair1['qz_a'], pz).sum(dim=1).mean()
                    + kl(out_unpair1['qz_b'], pz).sum(dim=1).mean()
                )
                
                loss = cont_loss + kl_loss * alpha
            
                loss.backward()
                opt.step()
                total_loss.append(loss)
            
        total_loss = sum(total_loss).log()
        scheduler.step(total_loss)
        
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history

##------------------------------------------------------------------------------------------------
## tri-modality integration
class trivib(torch.nn.Module):
    def __init__(self,
                 n_input_a=2000,
                 n_input_b=2000,
                 n_input_c=2000,
                 n_hidden=256, n_latent=10,
                 n_batch=1,
                 maskB=None, relationB='positive',
                 maskC=None, relationC='positive'):
        super(trivib, self).__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_input_c = n_input_c
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        
        self.maskedlinearB = MaskedLinear(self.n_input_b, self.n_input_a)
        if maskB is not None:
            self.maskedlinearB.set_mask(maskB)
        self.translatorB = torch.nn.Sequential(
            self.maskedlinearB,
            torch.nn.BatchNorm1d(self.n_input_a)
        )
        
        self.maskedlinearC = MaskedLinear(self.n_input_c, self.n_input_a)
        if maskC is not None:
            self.maskedlinearC.set_mask(maskC)
        self.translatorC = torch.nn.Sequential(
            self.maskedlinearC,
            torch.nn.BatchNorm1d(self.n_input_a)
        )
        
        self.encoderA = VariationalEncoder(
            n_input=self.n_input_a, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecterA = torch.nn.Linear(self.n_latent+self.n_batch, 128)
        
        self.apply(init_weights)
        
        if relationB=='positive':
            initial_weights = torch.ones((self.n_input_a, self.n_input_b)) # torch.eye(self.n_input)
        else:
            initial_weights = -torch.ones((self.n_input_a, self.n_input_b)) # torch.eye(self.n_input)
        
        if maskB is not None:
            initial_weights[maskB!=1]=0
        self.translatorB[0].weight.data = initial_weights.data.to(self.translatorB[0].weight.device, 
                                                                  self.translatorB[0].weight.dtype)
        
        if relationC=='positive':
            initial_weights = torch.ones((self.n_input_a, self.n_input_c)) # torch.eye(self.n_input)
        else:
            initial_weights = -torch.ones((self.n_input_a, self.n_input_c)) # torch.eye(self.n_input)
        
        if maskC is not None:
            initial_weights[maskC!=1]=0
        self.translatorC[0].weight.data = initial_weights.data.to(self.translatorC[0].weight.device, 
                                                                  self.translatorC[0].weight.dtype)

    def forward(self, x_a, x_b, x_c, batcha, batchb, batchc):
        
        x_BtoA = self.translatorB(x_b)
        qz_b, z_b = self.encoderA(x_BtoA)
        x_CtoA = self.translatorC(x_c)
        qz_c, z_c = self.encoderA(x_CtoA)
        qz_a, z_a = self.encoderA(x_a)
        
        z_a = self.transform(z_a)
        z_b = self.transform(z_b)
        z_c = self.transform(z_c)
        
        p_a = self.projecterA(torch.cat((z_a, batcha), 1))
        p_b = self.projecterA(torch.cat((z_b, batchb), 1))
        p_c = self.projecterA(torch.cat((z_c, batchc), 1))
            
        return {'z_a': z_a, 'z_b': z_b, 'z_c': z_c,
                'qz_a': qz_a, 'qz_b': qz_b, 'qz_c': qz_c,
                'proj_a': p_a, 'proj_b': p_b, 'proj_c': p_c}


class trivib_species(torch.nn.Module):
    def __init__(self,
                 n_input=2000,
                 n_input_a=2000,
                 n_input_b=2000,
                 n_input_c=2000,
                 n_hidden=256, n_latent=10,
                 n_batch=1,
                 maskA=None, relationA='positive',
                 maskB=None, relationB='positive',
                 maskC=None, relationC='positive'):
        super(trivib_species, self).__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_input_c = n_input_c
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        
        self.maskedlinearA = MaskedLinear(self.n_input_a, self.n_input)
        if maskA is not None:
            self.maskedlinearA.set_mask(maskA)
        self.translatorA = torch.nn.Sequential(
            self.maskedlinearA,
            torch.nn.BatchNorm1d(self.n_input),   
        )
        
        self.maskedlinearB = MaskedLinear(self.n_input_b, self.n_input)
        if maskB is not None:
            self.maskedlinearB.set_mask(maskB)
        self.translatorB = torch.nn.Sequential(
            self.maskedlinearB,
            torch.nn.BatchNorm1d(self.n_input),
        )
        
        self.maskedlinearC = MaskedLinear(self.n_input_c, self.n_input)
        if maskC is not None:
            self.maskedlinearC.set_mask(maskC)
        self.translatorC = torch.nn.Sequential(
            self.maskedlinearC,
            torch.nn.BatchNorm1d(self.n_input),
        )
        
        self.encoderA = VariationalEncoder(
            n_input=self.n_input, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecterA = torch.nn.Linear(self.n_latent+self.n_batch, self.n_hidden)
        
        self.apply(init_weights)
        
        if relationA=='positive':
            initial_weights = torch.ones((self.n_input, self.n_input_a))
        else:
            initial_weights = -torch.ones((self.n_input, self.n_input_a))
        
        if maskA is not None:
            initial_weights[maskA!=1]=0
        self.translatorA[0].weight.data = initial_weights.data.to(self.translatorA[0].weight.device, 
                                                                  self.translatorA[0].weight.dtype)
        
        if relationB=='positive':
            initial_weights = torch.ones((self.n_input, self.n_input_b))
        else:
            initial_weights = -torch.ones((self.n_input, self.n_input_b))
        
        if maskB is not None:
            initial_weights[maskB!=1]=0
        self.translatorB[0].weight.data = initial_weights.data.to(self.translatorB[0].weight.device, 
                                                                  self.translatorB[0].weight.dtype)
        
        if relationC=='positive':
            initial_weights = torch.ones((self.n_input, self.n_input_c))
        else:
            initial_weights = -torch.ones((self.n_input, self.n_input_c))
        
        if maskC is not None:
            initial_weights[maskC!=1]=0
        self.translatorC[0].weight.data = initial_weights.data.to(self.translatorC[0].weight.device, 
                                                                  self.translatorC[0].weight.dtype)

    def forward(self, x_a, x_b, x_c, batcha, batchb, batchc):
        
        x_BtoA = self.translatorB(x_b)
        qz_b, z_b = self.encoderA(x_BtoA)
        x_CtoA = self.translatorC(x_c)
        qz_c, z_c = self.encoderA(x_CtoA)
        x_AtoA = self.translatorA(x_a)
        qz_a, z_a = self.encoderA(x_AtoA)
        
        z_a = self.transform(z_a)
        z_b = self.transform(z_b)
        z_c = self.transform(z_c)
        
        p_a = self.projecterA(torch.cat((z_a, batcha), 1))
        p_b = self.projecterA(torch.cat((z_b, batchb), 1))
        p_c = self.projecterA(torch.cat((z_c, batchc), 1))
            
        return {'z_a': z_a, 'z_b': z_b, 'z_c': z_c,
                'qz_a': qz_a, 'qz_b': qz_b, 'qz_c': qz_c,
                'proj_a': p_a, 'proj_b': p_b, 'proj_c': p_c}

class trivibLoRA(torch.nn.Module):
    def __init__(self,
                 n_input=2000,
                 n_input_a=2000,
                 n_input_b=2000,
                 n_input_c=2000,
                 n_hidden=256, n_latent=10,
                 n_batch=1, rank=128):
        super(trivibLoRA, self).__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_input_c = n_input_c
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.rank = rank
        
        self.encoderA = VariationalEncoder(
            n_input=self.n_input, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecterA = torch.nn.Linear(self.n_latent+self.n_batch, self.n_hidden)
        
        self.apply(init_weights)
        
        self.lora_a = torch.nn.Linear(n_input_a, rank, bias=False)
        self.lora_b = torch.nn.Linear(n_input_b, rank, bias=False)
        self.lora_c = torch.nn.Linear(n_input_c, rank, bias=False)
        self.lora = torch.nn.Linear(rank, n_input, bias=False)
        self.batchnormA = torch.nn.BatchNorm1d(n_input)
        self.batchnormB = torch.nn.BatchNorm1d(n_input)
        self.batchnormC = torch.nn.BatchNorm1d(n_input)
        
        bias = torch.zeros(n_input)
        self.register_buffer('biasA', bias)
        self.register_buffer('biasB', bias)
        self.register_buffer('biasC', bias)
        
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, 
                                       mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.lora_b.weight, 
                                       mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.lora_c.weight, 
                                       mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.lora.weight, 
                                       mode='fan_in', nonlinearity='relu')
        
        torch.nn.init.normal_(self.batchnormA.weight, 1.0, 0.02)
        torch.nn.init.zeros_(self.batchnormA.bias)
        torch.nn.init.normal_(self.batchnormB.weight, 1.0, 0.02)
        torch.nn.init.zeros_(self.batchnormB.bias)
        torch.nn.init.normal_(self.batchnormC.weight, 1.0, 0.02)
        torch.nn.init.zeros_(self.batchnormC.bias)

    def forward(self, x_a, x_b, x_c, batcha, batchb, batchc):
        
        x_AtoA = self.lora_a(x_a)
        x_AtoA = self.lora(x_AtoA)
        x_AtoA = x_AtoA + self.biasA # * (32 / self.rank)
        x_AtoA = self.batchnormA(x_AtoA)
        qz_a, z_a = self.encoderA(x_AtoA)
        
        x_BtoA = self.lora_b(x_b)
        x_BtoA = self.lora(x_BtoA)
        x_BtoA = x_BtoA + self.biasB # * (32 / self.rank)
        x_BtoA = self.batchnormB(x_BtoA)
        qz_b, z_b = self.encoderA(x_BtoA)
        
        x_CtoA = self.lora_c(x_c)
        x_CtoA = self.lora(x_CtoA)
        x_CtoA = x_CtoA + self.biasC # * (32 / self.rank) 
        x_CtoA = self.batchnormC(x_CtoA)
        qz_c, z_c = self.encoderA(x_CtoA)
        
        z_a = self.transform(z_a)
        z_b = self.transform(z_b)
        z_c = self.transform(z_c)
        
        p_a = self.projecterA(torch.cat((z_a, batcha), 1))
        p_b = self.projecterA(torch.cat((z_b, batchb), 1))
        p_c = self.projecterA(torch.cat((z_c, batchc), 1))
            
        return {'z_a': z_a, 'z_b': z_b, 'z_c': z_c,
                'qz_a': qz_a, 'qz_b': qz_b, 'qz_c': qz_c,
                'proj_a': p_a, 'proj_b': p_b, 'proj_c': p_c}

def multivib_tritraining(model,
                         Xa, Xb, Xc,
                         batcha, batchb, batchc,
                         epoch=100, batch_size=128, 
                         temp=0.15, alpha=0.05, beta=0.0, l1_reg=False,
                         random_seed=0):
    
    contrastive_loss = DCL(temperature=temp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = VSGD(model.parameters(), lr=0.0006, ps=1e-7, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                           factor=0.8,
                                                           patience=20, 
                                                           min_lr=0.0001, 
                                                           verbose=True)
    
    loss_history = []
    for e in range(epoch):
            
        model.to(device)
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xa.shape[0])
        X_train_A = Xa[r,:]
        y_batch_A = batcha[r,:]
        X_tensor_A=torch.tensor(X_train_A).float()
        y_tensor_A=torch.tensor(y_batch_A).float()
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xb.shape[0])
        X_train_B = Xb[r,:]
        y_batch_B = batchb[r,:]
        X_tensor_B=torch.tensor(X_train_B).float()
        y_tensor_B=torch.tensor(y_batch_B).float()
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xc.shape[0])
        X_train_C = Xc[r,:]
        y_batch_C = batchc[r,:]
        X_tensor_C=torch.tensor(X_train_C).float()
        y_tensor_C=torch.tensor(y_batch_C).float()
        
        n = min(Xa.shape[0], Xb.shape[0], Xc.shape[0])
        
        total_loss = []
        
        with tqdm(total=n//batch_size, 
                  desc=f"Epoch {e+1}/{epoch}",
                  unit="batch",
                  bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False,
                  position=0) as pbar:
            
            for i in range(n//batch_size):
                
                pbar.update(1)
                
                opt.zero_grad()
                
                inputs_a1 = X_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_a2 = X_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_a = y_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_a1.shape
                inputs_a1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_a2 += torch.normal(0, 1.0, (c, m)).to(device)
            
                inputs_b1 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_b2 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_b = y_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_b1.shape
                inputs_b1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_b2 += torch.normal(0, 1.0, (c, m)).to(device)
                
                inputs_c1 = X_tensor_C[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_c2 = X_tensor_C[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_c = y_tensor_C[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_c1.shape
                inputs_c1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_c2 += torch.normal(0, 1.0, (c, m)).to(device)
                
                out_unpair1 = model(inputs_a1,inputs_b1,inputs_c1,batch_a,batch_b,batch_c)
                out_unpair2 = model(inputs_a2,inputs_b2,inputs_c2,batch_a,batch_b,batch_c)
            
                # Contrastive loss
                cont_loss = (
                    contrastive_loss(out_unpair1['proj_a'], out_unpair2['proj_a'])
                    + contrastive_loss(out_unpair1['proj_b'], out_unpair2['proj_b'])
                    + contrastive_loss(out_unpair1['proj_c'], out_unpair2['proj_c'])
                )
                
                # KL divergence
                pz = Normal(
                    torch.zeros_like(out_unpair1['qz_a'].mean),
                    torch.ones_like(out_unpair1['qz_a'].mean),
                )
            
                kl_loss = (
                    kl(out_unpair1['qz_a'], pz).sum(dim=1).mean()
                    + kl(out_unpair1['qz_b'], pz).sum(dim=1).mean()
                    + kl(out_unpair1['qz_c'], pz).sum(dim=1).mean()
                )
                
                loss = cont_loss + kl_loss * alpha
            
                loss.backward()
                opt.step()
                total_loss.append(loss)
            
        total_loss = sum(total_loss).log()
        scheduler.step(total_loss)
        
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history

def multivib_metrictraining(model,
                            Xa, Xb, Xc,
                            batcha, batchb, batchc,
                            epoch=100, batch_size=128, 
                            temp=0.15, alpha=0.05, beta=0.0, l1_reg=False,
                            random_seed=0):
    
    contrastive_loss = DCL(temperature=temp)
    TripletLoss = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = VSGD(model.parameters(), lr=0.0006, ps=1e-7, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                           factor=0.8,
                                                           patience=20, 
                                                           min_lr=0.0001, 
                                                           verbose=True)
    
    loss_history = []
    for e in range(epoch):
            
        model.to(device)
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xa.shape[0])
        X_train_A = Xa[r,:]
        y_batch_A = batcha[r,:]
        X_tensor_A=torch.tensor(X_train_A).float()
        y_tensor_A=torch.tensor(y_batch_A).float()
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xb.shape[0])
        X_train_B = Xb[r,:]
        y_batch_B = batchb[r,:]
        X_tensor_B=torch.tensor(X_train_B).float()
        y_tensor_B=torch.tensor(y_batch_B).float()
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xc.shape[0])
        X_train_C = Xc[r,:]
        y_batch_C = batchc[r,:]
        X_tensor_C=torch.tensor(X_train_C).float()
        y_tensor_C=torch.tensor(y_batch_C).float()
        
        n = min(Xa.shape[0], Xb.shape[0], Xc.shape[0])
        
        total_loss = []
        
        with tqdm(total=n//batch_size, 
                  desc=f"Epoch {e+1}/{epoch}",
                  unit="batch",
                  bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False,
                  position=0) as pbar:
            
            for i in range(n//batch_size):
                
                pbar.update(1)
                
                opt.zero_grad()
                
                inputs_a1 = X_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_a2 = X_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_a = y_tensor_A[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_a1.shape
                inputs_a1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_a2 += torch.normal(0, 1.0, (c, m)).to(device)
            
                inputs_b1 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_b2 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_b = y_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_b1.shape
                inputs_b1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_b2 += torch.normal(0, 1.0, (c, m)).to(device)
                
                inputs_c1 = X_tensor_C[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_c2 = X_tensor_C[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_c = y_tensor_C[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_c1.shape
                inputs_c1 += torch.normal(0, 1.0, (c, m)).to(device)
                inputs_c2 += torch.normal(0, 1.0, (c, m)).to(device)
                
                out_unpair1 = model(inputs_a1,inputs_b1,inputs_c1,batch_a,batch_b,batch_c)
                out_unpair2 = model(inputs_a2,inputs_b2,inputs_c2,batch_a,batch_b,batch_c)
            
                # Triplet loss
                ABcos = F.cosine_similarity(out_unpair1['z_a'].unsqueeze(1), out_unpair1['z_b'].unsqueeze(0), dim=2)
                ACcos = F.cosine_similarity(out_unpair1['z_a'].unsqueeze(1), out_unpair1['z_c'].unsqueeze(0), dim=2)
                BCcos = F.cosine_similarity(out_unpair1['z_b'].unsqueeze(1), out_unpair1['z_c'].unsqueeze(0), dim=2)
                
                ABneg = torch.argmax(ABcos, dim=1)
                BAneg = torch.argmax(ABcos, dim=0)
                ACneg = torch.argmax(ACcos, dim=1)
                CAneg = torch.argmax(ACcos, dim=0)
                BCneg = torch.argmax(BCcos, dim=1)
                CBneg = torch.argmax(BCcos, dim=0)
                
                triplet_loss = (
                    TripletLoss(out_unpair1['z_a'], out_unpair2['z_a'], out_unpair2['z_b'][ABneg,:]) 
                    + TripletLoss(out_unpair1['z_a'], out_unpair2['z_a'], out_unpair2['z_c'][ACneg,:])
                    + TripletLoss(out_unpair1['z_b'], out_unpair2['z_b'], out_unpair2['z_a'][BAneg,:])
                    + TripletLoss(out_unpair1['z_b'], out_unpair2['z_b'], out_unpair2['z_c'][BCneg,:])
                    + TripletLoss(out_unpair1['z_c'], out_unpair2['z_c'], out_unpair2['z_a'][CAneg,:])
                    + TripletLoss(out_unpair1['z_c'], out_unpair2['z_c'], out_unpair2['z_b'][CBneg,:])
                )
                
                # Contrastive loss
                cont_loss = (
                    contrastive_loss(out_unpair1['proj_a'], out_unpair2['proj_a'])
                    + contrastive_loss(out_unpair1['proj_b'], out_unpair2['proj_b'])
                    + contrastive_loss(out_unpair1['proj_c'], out_unpair2['proj_c'])
                )
                
                # KL divergence
                pz = Normal(
                    torch.zeros_like(out_unpair1['qz_a'].mean),
                    torch.ones_like(out_unpair1['qz_a'].mean),
                )
            
                kl_loss = (
                    kl(out_unpair1['qz_a'], pz).sum(dim=1).mean()
                    + kl(out_unpair1['qz_b'], pz).sum(dim=1).mean()
                    + kl(out_unpair1['qz_c'], pz).sum(dim=1).mean()
                )
                
                if l1_reg:
                    lasso_reg = (
                        torch.sum(torch.linalg.norm(model.lora_a.weight, ord=1, dim=1))
                        + torch.sum(torch.linalg.norm(model.lora_b.weight, ord=1, dim=1))
                        + torch.sum(torch.linalg.norm(model.lora_c.weight, ord=1, dim=1))
                    )
                    
                    # gene_pA = Normal(
                    #     torch.zeros_like(model.lora_a.weight),
                    #     torch.ones_like(model.lora_a.weight)
                    # )
                    # gene_pB = Normal(
                    #     torch.zeros_like(model.lora_b.weight),
                    #     torch.ones_like(model.lora_b.weight)
                    # )
                    # gene_pC = Normal(
                    #     torch.zeros_like(model.lora_c.weight),
                    #     torch.ones_like(model.lora_c.weight)
                    # )
                    # kl_reg = (
                    #     gene_pA.log_prob(model.lora_a.weight).sum(dim=1).mean()
                    #     + gene_pB.log_prob(model.lora_b.weight).sum(dim=1).mean()
                    #     + gene_pC.log_prob(model.lora_c.weight).sum(dim=1).mean()
                    # )
                    
                    loss = cont_loss + triplet_loss * 0.25 + kl_loss * alpha + lasso_reg * beta
                
                else:
                    loss = cont_loss + triplet_loss * 0.25 + kl_loss * alpha
            
                loss.backward()
                opt.step()
                total_loss.append(loss)
            
        total_loss = sum(total_loss).log()
        scheduler.step(total_loss)
        
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history
