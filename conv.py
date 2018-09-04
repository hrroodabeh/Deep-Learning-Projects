import numpy as np
import matplotlib.pyplot as plt
import h5py

def zero_padding(X, p):
    # X of [m, n_h, n_w, n_c] Dimensions
    # padding around each channel of size n_h*n_w with p zeros
    return np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant', constant_values=0)

def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev*W
    Z = np.sum(np.sum(s))
    Z = Z + b
    return Z


def conv_forward(A_prev, W, b, hparameters):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int((n_H_prev + 2*pad - f)/stride) + 1
    n_W = int((n_W_prev + 2*pad - f)/stride) + 1
    
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = zero_padding(A_prev, pad)
    
    for i in range(m):                             
        a_prev_pad = A_prev_pad[i]                  
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                   
                    
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
                                            
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache


def pool_forward(A_prev, hparameters, mode = "max"):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range (n_C):
                    
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.sum(np.sum(a_prev_slice))/(a_prev_slice.shape[0]*a_prev_slice.shape[1])

    cache = (A_prev, hparameters)
    
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache