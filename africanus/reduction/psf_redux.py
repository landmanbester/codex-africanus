#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import dask.array as da

iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def FFT(x):
    return da.fft.fftshift(da.fft.fft2(da.fft.ifftshift(x)))

def iFFT(x):
    return da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(x)))

def F(x):
    return Fs(np.fft.fft2(iFs(x)))


def iF(x):
    return Fs(np.fft.ifft2(iFs(x)))


def PSF_response(image, PSF_hat, Sigma):

    # im_pad = np.pad(image, padding, 'constant')
    im_hat = F(image)

    # apply element-wise product with PSF_hat for convolution
    vis = PSF_hat*im_hat

    # apply weights
    w_im = Sigma*vis.flatten()

    new_im = iF(w_im.reshape(im_hat.shape))

    return new_im


def PSF_adjoint(image, PSF_hat, Sigma):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis:
    :param PSF:
    :return:
    """

    im_hat = F(image).flatten()

    w_im = (Sigma * im_hat).reshape(image.shape)

    vis = PSF_hat.conj()*w_im

    new_im = iF(vis)

    return new_im


def diag_probe(A, dim, maxiter=2000, tol=1e-8, mode="Bernoulli"):

    D = np.zeros(dim, dtype='complex128')
    t = np.zeros(dim, dtype='complex128')
    q = np.zeros(dim, dtype='complex128')

    if mode == "Normal":
        gen_random = lambda dim: np.random.randn(dim) + 1.0j*np.random.randn(dim)
    elif mode == "Bernoulli":
        gen_random = lambda dim: np.where(np.random.random(dim) < 0.5, -1, 1) + \
                                 1.0j*np.where(np.random.random(dim) < 0.5, -1, 1)

    for k in range(maxiter):
        v = gen_random(dim)
        t += A(v)*v.conj()
        q += v*v.conj()
        D_new = t/q
        norm = np.linalg.norm(D_new)
        rel_norm = np.linalg.norm(D_new - D)/norm
        if k % 10 == 0:
            print("relative norm {0}: {1}".format(k, rel_norm))
        if rel_norm < tol:
            "Took {0} iterations to find the diagonal".format(k)
            return D
        D = D_new
    print("Final relative norm: ", rel_norm)
    return D


def guessmatrix(operator, M, N):
    '''
    Compute the covariance matrix by applying a given operator (F*Phi^T*Phi) on different delta functions
    '''
    from scipy.sparse import coo_matrix
    from scipy.sparse import csc_matrix

    #maxnonzeros = min(M, N)
    operdiag = np.zeros(N, dtype='complex')
    for i in np.arange(N):
        # deltacol = coo_matrix(([1], ([i], [0])), shape=(N, 1))
        deltacol = np.zeros((N, 1))
        deltacol[i] = 1.0
        deltacol = da.from_array(deltacol, chunks=(N, 1))
        currcol = operator(deltacol).flatten()
        #if i > maxnonzeros: break
        operdiag[i] = currcol[i]

    #matrix = coo_matrix((operdiag, (np.arange(maxnonzeros), np.arange(maxnonzeros))), shape=(M, N))

    return operdiag

if __name__=="__main__":
    # first construct a positive semi-definite operator
    Asqrt = np.random.randn(25, 10) + 1.0j*np.random.randn(25, 10)
    A = Asqrt.conj().T.dot(Asqrt)

    # get true diagonal
    diag_true = np.diag(A)

    # now get the diagonal via probing
    Aop = lambda x: A.dot(x)
    dim = A.shape[0]
    diag_estimate = diag_probe(Aop, dim)
    diag_estimate2 = guessmatrix(Aop, 10, 10)

    print A.shape
    print diag_estimate2

    x = np.linspace(0, 1.5*diag_true.max(), 100)
    import matplotlib.pyplot as plt
    plt.figure('diag')
    plt.plot(x, x, 'k')
    plt.plot(diag_true.real, diag_estimate.real, 'rx')
    plt.plot(diag_true.real, diag_estimate2.real, 'b+')
    plt.show()




