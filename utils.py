import os
from numpy.linalg import norm
import numpy as np
np.random.seed(1235)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_spectral(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight_u.data.normal_(0.0, 0.02)
        m.weight_v.data.normal_(0.0, 0.02)
        m.weight_bar.data.normal_(0.0, 0.02)

def mkdirp(path):
    try:
        os.mkdir(path)
    except:
        pass

def unitvec_like(vec):
    unitvec = np.random.randn(*vec.shape)
    return unitvec / norm(unitvec.ravel())

def get_randdir_filtnormed(weights, generator=False):
    # create random direction vectors in weight space filternormalized
    
    randdir = {}
    for key, layer in weights.items():
        
        # handle nonconvolutional layers
        if len(list(layer.shape)) != 4: randdir[key] = np.zeros(layer.shape); continue
        
        if generator: layer = layer.transpose(1,0)
        
        # make randdir filters have same norm as the corresponding filter in the weights
        layerR = np.array([ unitvec_like(filter) * norm(filter) for filter in layer.cpu().numpy() ])
        
        if generator: layerR = layerR.transpose(1,0,2,3)
        
        randdir[key] = layerR
    
    return randdir
