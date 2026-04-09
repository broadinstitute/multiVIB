import gc
import numpy as np
from tqdm import tqdm

from sklearn.utils import class_weight
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from .module import DCL, OODAlignmentLoss, GraphNeighborhoodReg, VICRegLoss
from .module import VariationalEncoder, MaskedLinear, LoRALinear, CellTypeClassifier


def crossover_augmentation(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Applies the CrossOver augmentation to a batch of single-cell gene expression data.
    
    Args:
        x (torch.Tensor): A batch of single-cell data, shape (batch_size, num_genes).
        alpha (float): The percentage of genes to mutate (e.g., 0.1 for 10% or 0.25 for 25%).
        
    Returns:
        torch.Tensor: The augmented batch with genes swapped from random cells.
    """
    batch_size, num_genes = x.shape
    
    # Randomly shuffle the batch to assign "another random cell" to each current cell
    shuffled_indices = torch.randperm(batch_size, device=x.device)
    x_random_cells = x[shuffled_indices]
    
    # Create a boolean mask to select 'alpha' proportion of genes to swap
    # shape: (batch_size, num_genes)
    swap_mask = torch.rand((batch_size, num_genes), device=x.device) < alpha
    
    # Clone the original data to create the augmented view
    x_augmented = x.clone()
    
    # Apply the mutation by replacing values where the mask is True
    x_augmented[swap_mask] = x_random_cells[swap_mask]
    
    return x_augmented

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
                 mask=None, joint=True, relation='positive'):
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
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, 64)
        
        self.apply(init_weights)
        
        if relation=='positive':
            initial_weights = torch.ones((self.n_input_a, self.n_input_b))
        else:
            initial_weights = -torch.ones((self.n_input_a, self.n_input_b))
        
        if mask is not None:
            initial_weights[mask!=1]=1e-6
            self.translator[0].weight.data = initial_weights.data.to(self.translator[0].weight.device, 
                                                                     self.translator[0].weight.dtype)
        
    def forward(self, x_a, x_b, batcha, batchb):
        
        x_BtoA = self.translator(x_b)
        qz_a, z_a = self.encoder(x_a)
        qz_b, z_b = self.encoder(x_BtoA)
        
        if self.joint:
            p_a = z_a
            p_b = z_b
            
        else:
            p_a = self.projecter(torch.cat((z_a, batcha), 1))
            p_b = self.projecter(torch.cat((z_b, batchb), 1))
            
        return {'z_a': z_a, 'z_b': z_b,
                'qz_a': qz_a, 'qz_b': qz_b,
                'proj_a': p_a, 'proj_b':p_b, 'a_trans': x_BtoA}
    
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
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, 64)
        
        self.apply(init_weights)
        
        self.translator = LoRALinear(self.n_input_b, self.n_input_a, self.rank)
        
    def forward(self, x_a, x_b, batcha, batchb):
        
        x_BtoA = self.translator(x_b)
        qz_b, z_b = self.encoder(x_BtoA)
        qz_a, z_a = self.encoder(x_a)
        
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
                               temp=0.15, alpha=0.05, beta=0.2,
                               crossover_rate=0.0, gaussian_rate_var=1.0,
                               random_seed=0, if_lr=False):
    
    if if_lr:
        # initialize translator with linear regression
        print('Initialization through Linear regression')
        lr = LinearRegression().fit(Xb_pair, Xa_pair)
        translator_weights = torch.from_numpy(lr.coef_)
        with torch.no_grad():
            model.translator[0].weight.copy_(translator_weights)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_loss = DCL(temperature=temp)
    ood_alignment = OODAlignmentLoss(
        n_prototypes=64,
        latent_dim=64,
        sinkhorn_eps=0.05,
        # mnn_temperature=0.15,
        # mnn_confidence=0.0,
        ot_weight=1.0,
        # pseudo_weight=1.0,
        cluster_momentum=0.99,
        use_pseudo_labels_for_ot=False, # True,
    )
    ood_alignment.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=5e-4)
    
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
                inputs_a1 = crossover_augmentation(inputs_a1, crossover_rate)
                inputs_a2 = crossover_augmentation(inputs_a2, crossover_rate)
                inputs_a1 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                inputs_a2 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
            
                inputs_b1 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_b2 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_b = y_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_b1.shape
                inputs_b1 = crossover_augmentation(inputs_b1, crossover_rate)
                inputs_b2 = crossover_augmentation(inputs_b2, crossover_rate)
                inputs_b1 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                inputs_b2 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
            
                inputs_apair = X_tensor_Apair[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_bpair = X_tensor_Bpair[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_apair = y_tensor_Apair[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_bpair = y_tensor_Bpair[i*batch_size:(i+1)*batch_size,:].to(device)
                
                model.joint = False
                out_unpair1 = model(inputs_a1, inputs_b1, batch_a, batch_b)
                out_unpair2 = model(inputs_a2, inputs_b2, batch_a, batch_b)
            
                model.joint = True
                out_pair = model(inputs_apair, inputs_bpair, 
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
                
                ood_loss, _ = ood_alignment(out_unpair1['proj_a'], 
                                            out_unpair1['proj_b'])
                
                loss = cont_loss + kl_loss * alpha + ood_loss * beta
                
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
                                 temp=0.15, alpha=0.05, beta=0.2,
                                 crossover_rate=0.0, gaussian_rate_var=1.0,
                                 random_seed=0):
    
    model.joint = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_loss = DCL(temperature=temp)
    ood_alignment = OODAlignmentLoss(
        n_prototypes=64,
        latent_dim=64,
        sinkhorn_eps=0.05,
        ot_weight=1.0,
        # pseudo_weight=1.0,
        cluster_momentum=0.99,
        use_pseudo_labels_for_ot=False, # True,
    )
    ood_alignment.to(device)
    graph_reg = GraphNeighborhoodReg(
        k = 15,
        weight_alignment = 0.5,     # coarse alignment
        weight_contrastive = 1.0,   # primary discriminability signal
        weight_laplacian = 0.5,     # spectral structure (scale down if unstable)
        weight_diffusion = 0.0,     # enable only for small batches
        contrastive_margin = 0.5,   # increase → more aggressive separation
        n_negative_samples = 64
    )
    graph_reg.to(device)
    vicreg = VICRegLoss()
    
    opt = torch.optim.AdamW(model.parameters(),
                            lr=0.0006, weight_decay=5e-4)
    
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
                inputs_a1 = crossover_augmentation(inputs_a1, crossover_rate)
                inputs_a2 = crossover_augmentation(inputs_a2, crossover_rate)
                inputs_a1 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                inputs_a2 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
            
                inputs_b1 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                inputs_b2 = X_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                batch_b = y_tensor_B[i*batch_size:(i+1)*batch_size,:].to(device)
                c, m = inputs_b1.shape
                inputs_b1 = crossover_augmentation(inputs_b1, crossover_rate)
                inputs_b2 = crossover_augmentation(inputs_b2, crossover_rate)
                inputs_b1 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                inputs_b2 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                
                out_unpair1 = model(inputs_a1,inputs_b1, batch_a, batch_b)
                out_unpair2 = model(inputs_a2,inputs_b2, batch_a, batch_b)
                
                # Contrastive loss
                cont_loss = (
                    contrastive_loss(out_unpair1['proj_a'], out_unpair2['proj_a'])
                    + contrastive_loss(out_unpair1['proj_b'], out_unpair2['proj_b']) * 2.0
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
                
                ood_loss, _ = ood_alignment(out_unpair1['proj_a'],
                                            out_unpair1['proj_b'])
                
                graph_loss = graph_reg(out_unpair1['proj_a'], out_unpair1['proj_b'])
                
                loss = cont_loss + kl_loss * alpha + ood_loss * beta + graph_loss
                loss += 0.1 * vicreg(out_unpair1['proj_a'], out_unpair1['proj_b'])
            
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
                 n_hidden=256, n_latent=10, n_batch=1, n_class=1):
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
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, 64)
        self.classifier = CellTypeClassifier(input_dim=self.n_latent, 
                                             num_classes=n_class)
        
        self.apply(init_weights)
        
        for i in range(len(mask)):
            if mask[i] is not None:
                if relations[i]=='positive':
                    initial_weights = torch.ones((self.n_shared_input, n_input[i]))
                else:
                    initial_weights = -torch.ones((self.n_shared_input, n_input[i]))
                initial_weights[mask!=1]=1e-6
                self.translator[i][0].weight.data = initial_weights.data.to(self.translator[i][0].weight.device, 
                                                                            self.translator[i][0].weight.dtype)

    def forward(self, xs , batches):
        
        z = []
        qz = []
        proj = []
        y = []
        for i in range(len(batches)):
            xt = self.translators[i](xs[i])
            qz_i, z_i = self.encoder(xt)
        
            p_i = self.projecter(torch.cat((z_i, batches[i]), 1))
            y_i = self.classifier(z_i)
            
            z.append(z_i)
            qz.append(qz_i)
            proj.append(p_i)
            y.append(y_i)
            
        return {'z': z, 'qz': qz, 'proj': proj, 'y': y}

class multivibLoRAS(torch.nn.Module):
    def __init__(self,
                 n_input=[2000, 2000, 2000],
                 n_shared_input=1000,
                 n_hidden=256, n_latent=10, n_batch=1, n_class=1, rank=128):
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
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, 64)
        self.classifier = CellTypeClassifier(input_dim=self.n_latent, 
                                             num_classes=n_class)
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
        y = []
        for i in range(len(batches)):
            xt_a = self.matrixA[i](xs[i])
            xt_b = self.matrixB(xt_a)
            xt = self.batchnorm(xt_b)
            qz_i, z_i = self.encoder(xt)
        
            p_i = self.projecter(torch.cat((z_i, batches[i]), 1))
            y_i = self.classifier(z_i)
            
            z.append(z_i)
            qz.append(qz_i)
            proj.append(p_i)
            y.append(y_i)
            
        return {'z': z, 'qz': qz, 'proj': proj, 'y': y}
    
def multivib_species_training(model,
                              Xs,
                              batches, cell_types,
                              epoch=100, batch_size=128, 
                              temp=0.15, alpha=0.05, beta=0.1,
                              param_setup='1st', 
                              crossover_rate=0.25, gaussian_rate_var=1.0,
                              random_seed=0):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_loss = DCL(temperature=temp)
    ood_alignment = OODAlignmentLoss(
        n_prototypes=64,
        latent_dim=64,
        sinkhorn_eps=0.05,
        ot_weight=1.0,
        # pseudo_weight=1.0,
        cluster_momentum=0.99,
        use_pseudo_labels_for_ot=False, # True,
    )
    ood_alignment.to(device)
    
    graph_reg = GraphNeighborhoodReg(
        k = 15,
        weight_alignment = 0.5,     # coarse alignment
        weight_contrastive = 1.0,   # primary discriminability signal
        weight_laplacian = 0.5,     # spectral structure (scale down if unstable)
        weight_diffusion = 0.5,     # enable only for small batches
        contrastive_margin = 0.5,   # increase → more aggressive separation
        n_negative_samples = 64
    )
    graph_reg.to(device)
    vicreg = VICRegLoss()
    
    ct = []
    for c in cell_types:
        ct += c
    ct = np.asarray(ct)
    ct_encoder = LabelEncoder()
    ct_encoder.fit(ct)
    encoded_ct = ct_encoder.transform(ct)
    unknown_class = encoded_ct[ct=='Unknown'][0]
    
    classes = np.unique(encoded_ct)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=classes,
        y=encoded_ct
    )
    class_weights[classes==unknown_class]=0.0
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    cls_criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    
    params = [{"params": model.parameters()}]
    
    if param_setup=='1st':
        for t in model.translators:
            params += [{"params": t.parameters()}]
    elif param_setup=='2nd':
        for t in model.matrixA:
            params += [{"params": t.parameters()}]
        
    opt = torch.optim.AdamW(params, lr=0.0006, weight_decay=5e-4)
    
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
        ct_tensor = []
        for i in range(len(Xs)):
            r = np.random.RandomState(seed=random_seed+e).permutation(Xs[i].shape[0])
            X_train = Xs[i][r,:]
            y_batch = batches[i][r,:]
            ct_batch = np.asarray(cell_types[i])[r]
            X_tensor.append(torch.tensor(X_train).float())
            y_tensor.append(torch.tensor(y_batch).float())
            ct_tensor.append(torch.tensor(ct_encoder.transform(ct_batch), 
                                          dtype=torch.long))
        
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
                ct_batch = []
                for j in range(num_species):
                    inputs_a1 = X_tensor[j][i*batch_size:(i+1)*batch_size,:].to(device)
                    inputs_a2 = X_tensor[j][i*batch_size:(i+1)*batch_size,:].to(device)
                    b = y_tensor[j][i*batch_size:(i+1)*batch_size,:].to(device)
                    ct = ct_tensor[j][i*batch_size:(i+1)*batch_size].to(device)
                    c, m = inputs_a1.shape
                    
                    inputs_a1 = crossover_augmentation(inputs_a1, crossover_rate)
                    inputs_a2 = crossover_augmentation(inputs_a2, crossover_rate)
                    inputs_a1 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                    inputs_a2 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                    inputs1.append(inputs_a1)
                    inputs2.append(inputs_a2)
                    batch.append(b)
                    ct_batch.append(ct)
                
                out_unpair1 = model(inputs1, batch)
                out_unpair2 = model(inputs2, batch)
                
                pz = Normal(
                    torch.zeros_like(out_unpair1['qz'][0].mean),
                    torch.ones_like(out_unpair1['qz'][0].mean),
                )
            
                kl_loss = kl(out_unpair1['qz'][0], pz).sum(dim=1).mean()
                cont_loss = contrastive_loss(out_unpair1['proj'][0], out_unpair2['proj'][0])
                
                if ct_batch[0][ct_batch[0]!=unknown_class].shape[0]>0:
                    clf_loss = cls_criterion(out_unpair1['y'][0][ct_batch[0]!=unknown_class,:], 
                                             ct_batch[0][ct_batch[0]!=unknown_class])
                    loss = cont_loss + kl_loss * alpha + clf_loss
                else:
                    loss = cont_loss + kl_loss * alpha
                
                for s in range(1, num_species):
                    cont_loss = contrastive_loss(out_unpair1['proj'][s], out_unpair2['proj'][s])
                    kl_loss = kl(out_unpair1['qz'][s], pz).sum(dim=1).mean()
                    
                    ood_loss, _ = ood_alignment(out_unpair1['proj'][s], 
                                                out_unpair1['proj'][s-1])
                    graph_loss = graph_reg(out_unpair1['proj'][s], 
                                           out_unpair1['proj'][s-1])
                    vic_reg = 0.1 * vicreg(out_unpair1['proj'][s], 
                                           out_unpair1['proj'][s-1])
                    
                    if ct_batch[s][ct_batch[s]!=unknown_class].shape[0]>0:
                        clf_loss = cls_criterion(out_unpair1['y'][s][ct_batch[s]!=unknown_class,:], 
                                                 ct_batch[s][ct_batch[s]!=unknown_class])
                        loss += cont_loss + kl_loss * alpha + ood_loss * beta + clf_loss + graph_loss + vic_reg
                    else:
                        loss += cont_loss + kl_loss * alpha + ood_loss * beta + graph_loss + vic_reg
                        
                loss.backward()
                opt.step()
                total_loss.append(loss)
            
        total_loss = sum(total_loss).log()
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history


##-----------------------------------------------------------------------------
## single modality only integration
class multivibR(torch.nn.Module):
    def __init__(self,
                 n_input_a=2000,
                 n_hidden=256, n_latent=10, n_batch=1, n_class=10,
                 mask=None, joint=True):
        super(multivibR, self).__init__()
        self.n_input_a = n_input_a
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_class = n_class
        
        self.encoder = VariationalEncoder(
            n_input=self.n_input_a, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent
        )
        self.projecter = torch.nn.Linear(self.n_latent+self.n_batch, 64)
        self.classifier = CellTypeClassifier(input_dim=self.n_latent, 
                                             num_classes=n_class)
        
        self.apply(init_weights)
        
    def forward(self, x_a, batcha):
        
        qz_a, z_a = self.encoder(x_a)
        p_a = self.projecter(torch.cat((z_a, batcha), 1))
        y_a = self.classifier(z_a)
        
        return {'z_a': z_a,
                'qz_a': qz_a,
                'proj_a': p_a, 'y_a': y_a}

def multivibR_training(model, Xa, batcha, cell_types,
                       epoch=100, batch_size=128, 
                       temp=0.15, alpha=0.05, 
                       crossover_rate=0.25, gaussian_rate_var=1.0,
                       random_seed=0):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_loss = DCL(temperature=temp)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=5e-4)
    
    ct_encoder = LabelEncoder()
    encoded_ct = ct_encoder.fit_transform(cell_types)
    unknown_class = encoded_ct[cell_types=='Unknown'][0]
    
    classes = np.unique(encoded_ct)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=classes,
        y=encoded_ct
    )
    class_weights[classes==unknown_class]=0.0
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    # cls_criterion = torch.nn.NLLLoss(weight=weights_tensor.to(device))
    cls_criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    
    loss_history = []
    for e in range(epoch):
            
        model.to(device)
        
        r = np.random.RandomState(seed=random_seed+e).permutation(Xa.shape[0])
        X_train_A = Xa[r,:]
        y_batch_A = batcha[r,:]
        cell_type_A = encoded_ct[r]
        X_tensor_A=torch.tensor(X_train_A).float()
        y_tensor_A=torch.tensor(y_batch_A).float()
        ct_tensor_A= torch.tensor(cell_type_A, dtype=torch.long)
        
        # if e%10 == 0 and e >= 50:
        #     with torch.no_grad():
        #         output = model(torch.tensor(Xa).float().to(device), 
        #                        torch.tensor(batcha).float().to(device))
        #         # ct_tensor= torch.argmax(F.softmax(output['y_a'], dim=1), 1)
        #         neigh = KNeighborsClassifier(n_neighbors=5)
        #         neigh.fit(output['qz_a'].mean.cpu().numpy()[cell_types!='Unknown',:],
        #                   encoded_ct[cell_types!='Unknown'])
        #         pred_ann = neigh.predict(output['qz_a'].mean.cpu().numpy())
        #         ct_tensor= torch.tensor(pred_ann)
        # if e >= 50:
        #     ct_tensor_A = ct_tensor[r].long()
        
        n = Xa.shape[0]
        
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
                ct_a = ct_tensor_A[i*batch_size:(i+1)*batch_size].to(device)
                c, m = inputs_a1.shape
                
                inputs_a1 = crossover_augmentation(inputs_a1, crossover_rate)
                inputs_a2 = crossover_augmentation(inputs_a2, crossover_rate)
                inputs_a1 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                inputs_a2 += torch.normal(0, gaussian_rate_var, (c, m)).to(device)
                
                out_unpair1 = model(inputs_a1, batch_a)
                out_unpair2 = model(inputs_a2, batch_a)
            
                # Contrastive loss
                cont_loss = contrastive_loss(out_unpair1['proj_a'], out_unpair2['proj_a'])
                
                # KL divergence
                pz = Normal(
                    torch.zeros_like(out_unpair1['qz_a'].mean),
                    torch.ones_like(out_unpair1['qz_a'].mean),
                )
                kl_loss = kl(out_unpair1['qz_a'], pz).sum(dim=1).mean()
                
                if ct_a[ct_a!=unknown_class].shape[0]>0:
                    clf_loss = cls_criterion(out_unpair1['y_a'][ct_a!=unknown_class,:], 
                                             ct_a[ct_a!=unknown_class])
                    # log_prob = torch.log(F.softmax(out_unpair1['y_a']/20.0, dim=1))
                    # clf_loss = cls_criterion(log_prob[ct_a!=unknown_class,:], 
                    #                          ct_a[ct_a!=unknown_class])
                    loss = cont_loss + kl_loss * alpha + clf_loss
                else:
                    loss = cont_loss + kl_loss * alpha
                
                loss.backward()
                opt.step()
                total_loss.append(loss)
            
        total_loss = sum(total_loss).log()
        loss_history.append(total_loss.cpu().detach().numpy())
        
    return loss_history
