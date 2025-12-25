import gc
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import torch
import torch.nn.functional as F
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

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""

    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

def scale_by_batch(x, batch_label):
    scaled_x = np.zeros_like(x)
    batch = list(set(batch_label))
    
    for b in batch:
        scaler = StandardScaler()
        scaled_x[batch_label==b,:] = scaler.fit_transform(x[batch_label==b,:])
    
    return scaled_x

##-----------------------------------------------------------------------------
## multivib model backbone
class multivib(torch.nn.Module):
    def __init__(self,
                 n_input_a=2000, n_input_b=2000, 
                 n_hidden=256, n_latent=10, n_batch=1, 
                 mask=None, joint=True):
        super(multivib, self).__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.joint = joint
        
        self.maskedlinear = MaskedLinear(self.n_input_b, self.n_input_a)
        if mask is not None:
            self.maskedlinear.set_mask(mask)
        self.translator = torch.nn.Sequential(
            self.maskedlinear,
            torch.nn.BatchNorm1d(self.n_input_a)
        )
        
        self.encoder = VariationalEncoder(
            n_input=self.n_input_a, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, 128)
        
        self.apply(init_weights)
        
    def forward(self, x_a, x_b, batcha, batchb):
        
        x_BtoA = self.translator(x_b)
        qz_a, z_a = self.encoder(x_a)
        qz_b, z_b = self.encoder(x_BtoA)
        
        z_a = self.transform(z_a)
        z_b = self.transform(z_b)
        
        if self.joint:
            p_a = z_a
            p_b = z_b
            
        else:
            p_a = self.projecter(torch.cat((z_a, batcha), 1))
            p_b = self.projecter(torch.cat((z_b, batchb), 1))
            
        return {'z_a': z_a, 'z_b': z_b,
                'qz_a': qz_a, 'qz_b': qz_b,
                'proj_a': p_a, 'proj_b':p_b}
    
class multivibLoRA(torch.nn.Module):
    def __init__(self,
                 n_input_a=2000, n_input_b=2000, 
                 n_hidden=256, n_latent=10, n_batch=1, rank=128,
                 joint=True):
        super(multivibLoRA, self).__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.rank = rank
        self.joint = joint
        
        self.encoder = VariationalEncoder(
            n_input=self.n_input_a, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, 128)
        
        self.apply(init_weights)
        
        self.translator = LoRALinear(self.n_input_b, self.n_input_a, self.rank)
        
    def forward(self, x_a, x_b, batcha, batchb):
        
        x_BtoA = self.translator(x_b)
        qz_b, z_b = self.encoder(x_BtoA)
        qz_a, z_a = self.encoder(x_a)
        
        z_a = self.transform(z_a)
        z_b = self.transform(z_b)
        
        if self.joint:
            p_a = z_a
            p_b = z_b
            
        else:
            p_a = self.projecter(torch.cat((z_a, batcha), 1))
            p_b = self.projecter(torch.cat((z_b, batchb), 1))
            
        return {'z_a': z_a, 'z_b': z_b,
                'qz_a': qz_a, 'qz_b': qz_b,
                'proj_a': p_a, 'proj_b':p_b}
    
def multivib_vertical_training(model,
                               Xa, Xb, Xa_pair, Xb_pair,
                               batcha, batchb, batcha_pair, batchb_pair,
                               epoch=100, batch_size=128, 
                               temp=0.15, alpha=0.05,
                               random_seed=0, if_lr=False):
    
    if if_lr:
        # initialize translator with linear regression
        print('Initialization through Linear regression')
        lr = LinearRegression().fit(Xb_pair, Xa_pair)
        translator_weights = torch.from_numpy(lr.coef_)
        with torch.no_grad():
            model.translator[0].weight.copy_(translator_weights)
            
    contrastive_loss = DCL(temperature=temp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = VSGD(model.parameters(), lr=0.0006, ps=1e-7, weight_decay=5e-4)
    
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
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xa_pair.shape[0])
        X_train_A = Xa_pair[r,:]
        y_batch_A = batcha_pair[r,:]
        X_tensor_Apair=torch.tensor(X_train_A).float()
        y_tensor_Apair=torch.tensor(y_batch_A).float()
        
        X_train_B = Xb_pair[r,:]
        y_batch_B = batchb_pair[r,:]
        X_tensor_Bpair=torch.tensor(X_train_B).float()
        y_tensor_Bpair=torch.tensor(y_batch_B).float()
        
        n = min(Xa.shape[0], Xb.shape[0], Xa_pair.shape[0])
        
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
            
                inputs_apair = X_tensor_Apair[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_bpair = X_tensor_Bpair[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_apair = y_tensor_Apair[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_bpair = y_tensor_Bpair[i*batch_size:(i+1)*batch_size,:].to(device)
                
                model.joint = False
                out_unpair1 = model(inputs_a1,inputs_b1, batch_a, batch_b)
                out_unpair2 = model(inputs_a2,inputs_b2, batch_a, batch_b)
            
                model.joint = True
                out_pair = model(inputs_apair,inputs_bpair, 
                                 batch_apair, batch_bpair)
                
                # Contrastive loss
                cont_loss = (
                    contrastive_loss(out_pair['proj_a'], out_pair['proj_b'])
                    + contrastive_loss(out_unpair1['proj_a'], out_unpair2['proj_a'])
                    + contrastive_loss(out_unpair1['proj_b'], out_unpair2['proj_b'])
                )
            
                # KL divergence
                pz = Normal(
                    torch.zeros_like(out_pair['qz_a'].mean),
                    torch.ones_like(out_pair['qz_a'].mean),
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
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history

def multivib_horizontal_training(model,
                                 Xa, Xb,
                                 batcha, batchb,
                                 epoch=100, batch_size=128, 
                                 temp=0.15, alpha=0.05,
                                 random_seed=0):
    
    model.joint = False
    contrastive_loss = DCL(temperature=temp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = VSGD(model.parameters(), lr=0.0006, ps=1e-7, weight_decay=5e-4)
    
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
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history


##-----------------------------------------------------------------------------
## multivib model backbone for cross-species integration
class multivibS(torch.nn.Module):
    def __init__(self,
                 n_input=[2000, 2000, 2000],
                 n_shared_input=1000,
                 masks=[None, None, None],
                 relations=['positive', 'positive', 'positive'],
                 n_hidden=256, n_latent=10, n_batch=1):
        super(multivibS, self).__init__()
        self.n_input = n_input
        self.n_shared_input = n_shared_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        
        self.translators = []
        for i in range(len(n_input)):
            maskedlinear = MaskedLinear(n_input[i], self.n_shared_input)
            if relations[i]=='positive':
                initial_weights = torch.ones((self.n_shared_input, n_input[i]))
            else:
                initial_weights = -torch.ones((self.n_shared_input, n_input[i]))
            if masks[i] is not None:
                maskedlinear.set_mask(masks[i])
                initial_weights[masks[i]!=1]=0
                maskedlinear.weight.data = initial_weights.data.to(maskedlinear.weight.device, 
                                                                   maskedlinear.weight.dtype)
            self.translators.append(
                torch.nn.Sequential(
                    maskedlinear,
                    torch.nn.BatchNorm1d(self.n_shared_input)
                )
            )
        
        self.encoder = VariationalEncoder(
            n_input=self.n_shared_input, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, self.n_hidden)
        
        self.apply(init_weights)

    def forward(self, xs , batches):
        
        z = []
        qz = []
        proj = []
        for i in range(len(batches)):
            xt = self.translators[i](xs[i])
            qz_i, z_i = self.encoder(xt)
        
            z_i = self.transform(z_i)
            p_i = self.projecter(torch.cat((z_i, batches[i]), 1))
            
            z.append(z_i)
            qz.append(qz_i)
            proj.append(p_i)
            
        return {'z': z, 'qz': qz, 'proj': proj}

class multivibLoRAS(torch.nn.Module):
    def __init__(self,
                 n_input=[2000, 2000, 2000],
                 n_shared_input=1000,
                 n_hidden=256, n_latent=10, n_batch=1, rank=128):
        super(multivibLoRAS, self).__init__()
        self.n_input = n_input
        self.n_shared_input = n_shared_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.rank = rank
        
        self.encoder = VariationalEncoder(
            n_input=self.n_shared_input, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.transform = torch.nn.Tanh()
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, self.n_hidden)
        
        self.apply(init_weights)
        
        self.matrixA = [torch.nn.Linear(n_input[i], self.rank, bias=False) for i in range(len(n_input))]
        self.matrixB = torch.nn.Linear(self.rank, self.n_shared_input, bias=True)
        self.batchnorm = torch.nn.BatchNorm1d(self.n_shared_input)
        
        for i in range(len(n_input)):
            torch.nn.init.kaiming_uniform_(self.matrixA[i].weight, 
                                           mode='fan_in', nonlinearity='relu')
        
        torch.nn.init.kaiming_uniform_(self.matrixB.weight, 
                                       mode='fan_in', nonlinearity='relu')
        torch.nn.init.normal_(self.batchnorm.weight, 1.0, 0.02)
        torch.nn.init.zeros_(self.batchnorm.bias)
        
    def forward(self, xs, batches):
        
        z = []
        qz = []
        proj = []
        for i in range(len(batches)):
            xt_a = self.matrixA[i](xs[i])
            xt_b = self.matrixB(xt_a)
            xt = self.batchnorm(xt_b)
            qz_i, z_i = self.encoder(xt)
        
            z_i = self.transform(z_i)
            p_i = self.projecter(torch.cat((z_i, batches[i]), 1))
            
            z.append(z_i)
            qz.append(qz_i)
            proj.append(p_i)
            
        return {'z': z, 'qz': qz, 'proj': proj}
    
def multivib_species_training(model,
                              Xs,
                              batches,
                              epoch=100, batch_size=128, 
                              temp=0.15, alpha=0.05, param_setup='1st',
                              random_seed=0):
    
    contrastive_loss = DCL(temperature=temp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = [{"params": model.parameters()}]
    
    if param_setup=='1st':
        for t in model.translators:
            params += [{"params": t.parameters()}]
    elif param_setup=='2nd':
        for t in model.matrixA:
            params += [{"params": t.parameters()}]
        
    opt = VSGD(params, lr=0.0006, ps=1e-7, weight_decay=5e-4)
    
    n = min([x.shape[0] for x in Xs])
    num_species = len(Xs)
    
    model.to(device)
    for i in range(num_species):
        if param_setup=='1st':
            model.translators[i].to(device)
        elif param_setup=='2nd':
            model.matrixA[i].to(device)
    
    loss_history = []
    for e in range(epoch):
        
        X_tensor = []
        y_tensor = []
        for i in range(len(Xs)):
            r = np.random.RandomState(seed=random_seed+e).permutation(Xs[i].shape[0])
            X_train = Xs[i][r,:]
            y_batch = batches[i][r,:]
            X_tensor.append(torch.tensor(X_train).float())
            y_tensor.append(torch.tensor(y_batch).float())
        
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
                
                inputs1 = []
                inputs2 = []
                batch = []
                for j in range(num_species):
                    inputs_a1 = X_tensor[j][i*batch_size:(i+1)*batch_size,:].to(device)
                    inputs_a2 = X_tensor[j][i*batch_size:(i+1)*batch_size,:].to(device)
                    b = y_tensor[j][i*batch_size:(i+1)*batch_size,:].to(device)
                    c, m = inputs_a1.shape
                    inputs_a1 += torch.normal(0, 1.0, (c, m)).to(device)
                    inputs_a2 += torch.normal(0, 1.0, (c, m)).to(device)
                    inputs1.append(inputs_a1)
                    inputs2.append(inputs_a2)
                    batch.append(b)
                
                out_unpair1 = model(inputs1, batch)
                out_unpair2 = model(inputs2, batch)
                
                pz = Normal(
                    torch.zeros_like(out_unpair1['qz'][0].mean),
                    torch.ones_like(out_unpair1['qz'][0].mean),
                )
            
                kl_loss = kl(out_unpair1['qz'][0], pz).sum(dim=1).mean()
                cont_loss = contrastive_loss(out_unpair1['proj'][0], out_unpair2['proj'][0])
                for s in range(1, num_species):
                    cont_loss += contrastive_loss(out_unpair1['proj'][s], out_unpair2['proj'][s])
                    kl_loss += kl(out_unpair1['qz'][s], pz).sum(dim=1).mean()
                
                loss = cont_loss + kl_loss * alpha
            
                loss.backward()
                opt.step()
                total_loss.append(loss)
            
        total_loss = sum(total_loss).log()
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history
