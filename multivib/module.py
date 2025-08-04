import torch
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_by_batch(x, batch_label):
    scaled_x = np.zeros_like(x)
    batch = list(set(batch_label))
    
    for b in batch:
        scaler = StandardScaler()
        scaled_x[batch_label==b,:] = scaler.fit_transform(x[batch_label==b,:])
    
    return scaled_x

def grad_norm(losses, model, gamma=1.0):
    
    # Compute gradients for each loss
    grads = []
    for loss in losses:
        loss.backward(retain_graph=True)
        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        grads.append(grad)
    
    # Compute GradNorm weights
    norms = torch.stack([torch.norm(g) for g in grads])
    avg_norm = torch.mean(norms)
    weights = (norms / avg_norm) ** -gamma
    
    return weights.detach()

class MaskedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mask_init_value=1.0):
        super().__init__(in_features, out_features, bias)
        mask_shape = (out_features, in_features)
        initial_mask = torch.full(mask_shape, mask_init_value)
        self.register_buffer('mask', initial_mask)
        
    def set_mask(self, mask):
        if self.mask.shape != mask.shape:
            raise ValueError(f"Mask shape mismatch. Expected {self.mask.shape}, got {mask.shape}")
        self.mask.data = mask.data.to(self.mask.device, self.mask.dtype)

    def get_masked_weight(self):
        return self.weight * self.mask

    def forward(self, x):
        # Apply mask to weights before the linear operation
        masked_weight = self.get_masked_weight()
        return F.linear(x, masked_weight, self.bias)
    
class LoRALinear(torch.nn.Module):
    def __init__(self, 
                 in_dim, out_dim, 
                 rank, # alpha, 
                 dropout=0.0, use_bias=False):
        
        super(LoRALinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        # self.alpha = alpha
        
        # LoRA A and B matrices
        self.lora_a = torch.nn.Linear(in_dim, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, out_dim, bias=False)
        self.batchnorm = torch.nn.BatchNorm1d(out_dim)
        
        bias = torch.zeros(out_dim)
        self.register_buffer('bias', bias)

        # Initialization of LoRA matrices
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, 
                                       mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.lora_b.weight, 
                                       mode='fan_in', nonlinearity='relu')
        
        torch.nn.init.normal_(self.batchnorm.weight, 1.0, 0.02)
        torch.nn.init.zeros_(self.batchnorm.bias)

    def forward(self, x):
        
        # LoRA forward pass
        lora_output = self.lora_a(x)
        lora_output = self.lora_b(lora_output)
        lora_output = lora_output + self.bias # * (self.alpha / self.rank) 
        lora_output = self.batchnorm(lora_output)

        return lora_output

class VariationalEncoder(torch.nn.Module):
    def __init__(self, n_input=2000,
                 n_hidden=128, 
                 n_latent=10,
                 var_eps=1e-4):
        super(VariationalEncoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.var_eps = var_eps
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
        )
        self.mean_encoder = torch.nn.Linear(n_hidden, n_latent)
        self.var_encoder = torch.nn.Linear(n_hidden, n_latent)
        
    def forward(self, x):
        q = self.encoder(x)
        qm = self.mean_encoder(q)
        qv = torch.exp(self.var_encoder(q)) + self.var_eps
        dist = Normal(qm, qv.sqrt())
        latent = dist.rsample()
        return dist, latent
    
class Encoder(torch.nn.Module):
    def __init__(self, n_input=2000, n_hidden=256, n_latent=10):
        super(Encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
        )
        self.z_encoder = torch.nn.Linear(n_hidden, n_latent)
        
    def forward(self, x):
        q = self.encoder(x)
        latent = self.z_encoder(q)
        return latent
    
class Decoder(torch.nn.Module):
    def __init__(self, n_output=2000,
                 n_hidden=128, 
                 n_latent=10,
                 var_eps=1e-4):
        super(Decoder, self).__init__()
        self.n_output = n_output
        self.n_latent = n_latent
        self.var_eps = var_eps
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent, n_hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
        )
        self.mean_encoder = torch.nn.Linear(n_hidden, n_output)
        self.var_encoder = torch.nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        q = self.encoder(x)
        qm = self.mean_encoder(q)
        qv = torch.exp(self.var_encoder(q)) + self.var_eps
        dist = Normal(qm, qv.sqrt())
        return dist


SMALL_NUM = np.log(1e-45)

class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        cross_view_distance = torch.mm(z1, z2.t())
        
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        
        return (positive_loss + negative_loss).mean()


class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)

        
# merge two fully connected layers
def merge_fc_layers(fc1, fc2):
    # Verify output dimensions match
    assert fc1.out_features == fc2.out_features, "Output dimensions must match"
    
    # Extract weights and biases
    W1, b1 = fc1.weight.data, fc1.bias.data
    W2, b2 = fc2.weight.data, fc2.bias.data
    
    # Concatenate weights horizontally and sum biases
    W_merged = torch.cat([W1, W2], dim=1)
    b_merged = b1 + b2
    
    # Create merged layer
    merged_fc = nn.Linear(W_merged.shape[1], W_merged.shape[0])
    merged_fc.weight.data = W_merged
    merged_fc.bias.data = b_merged
    
    return merged_fc