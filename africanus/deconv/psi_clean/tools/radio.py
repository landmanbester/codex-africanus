'''
Created on Nov 24, 2017

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np
import scipy.fftpack as scifft

##### Simulation parameters #####
class sparam:
    '''
    Simulation parameters
    '''
    def __init__(self, N=256, p=0.5, hole_number=1000, hole_prob=0.1, hole_size=np.pi/60, 
                 fpartition=np.array([-np.pi,np.pi]), sigma=np.pi/4, sigma_holes=np.pi/3):
        self.N = N  # number of pixels
        self.p = p  # ratio of visibility to number of pixels
        self.hole_number = hole_number # number of holes to introduce for "gaussian holes"
        self.hole_prob = hole_prob # probability of single pixel hole for "gaussian + missing pixels"
        self.hole_size = hole_size # size of the missing frequency data
        self.fpartition = fpartition # partition (symetrically) of the data to nodes (frequency ranges)
        self.sigma = sigma # variance of the gaussion over continous frequency
        self.sigma_holes = sigma_holes # variance of the gaussion for the holes

##### L2-ball parameters #####
class l2param:
    '''
    l2 ball parameters
    
    signature of the initialization 
            l2param(l2_ball_definition, stopping_criterion, bound, stop)
            
    l2_ball_definition:
        1 - 'value'
        2 - 'sigma'
        3 - 'chi-percentile'
    stopping_criterion:
        1 - 'l2-ball-percentage'
        2 - 'sigma'
        3 - 'chi-percentile' 
    bound, stop are paramters corresponding to l2_ball_definition and stopping_criterion respectively
    '''
    
    def __init__(self, l2_ball_definition, stopping_criterion, bound=0.99, stop=1.0001):
        self.l2_ball_definition = l2_ball_definition
        self.stopping_criterion = stopping_criterion
        self.bound = bound
        self.stop = stop


def simulateCoverage(num_pxl, num_vis, pattern='gaussian', holes=True):
    '''
    Simulate uv coverage
    '''
    
    sprm = sparam()
    sprm.N = num_pxl
    sprm.p = np.ceil(np.double(num_vis)/num_pxl)        
    u, v = util_gen_sampling_pattern(pattern, holes, sprm)
         
    return u, v  

def util_gen_sampling_pattern(pattern, holes, sprm):
    '''
    Generate a sampling pattern according to the given simulation parameter. 
    This function is also valid to generate a sampling pattern with holes.
    
    @param pattern: 'gaussian', 'uniform' or 'geometric'
    
    @param holes: boolean, whether to generate holes in the pattern
    
    @param sprm: object of parameters for the simulation, more details to see the class "sparam"
    
    @return: (u,v) sampling pattern
    
    '''
    
    from scipy.stats import norm 
    
    # Check simulation parameters
    if not hasattr(sprm, 'N'):
        sprm.N = 256
        print('Default the number of pixels is set to ' + str(sprm.N))
    if not hasattr(sprm, 'p'):
        sprm.p = 2
        print('Default ratio of visibility to number of pixels is set to ' + str(sprm.p))
    if not hasattr(sprm, 'fpartition'):
        sprm.fpartition = np.array([np.pi])
        print('Default symmetric partition is set to ' + str(sprm.fpartition))
    if not hasattr(sprm, 'hole_number'):
        sprm.hole_number = 10
        print('Default the number of holes is set to ' + str(sprm.hole_number))
    if not hasattr(sprm, 'sigma_holes'):
        sprm.sigma_holes = np.pi/2
        print('Default sigma of holes is set to ' + str(sprm.sigma_holes))
    if not hasattr(sprm, 'hole_size'):
        sprm.hole_size = np.pi/8
        print('Default the size of holes is set to ' + str(sprm.hole_size))
    if not hasattr(sprm, 'sigma'):
        sprm.sigma = np.pi/12
        print('Default sigma of the distribution is set to ' + str(sprm.sigma))

    
    Nh = sprm.hole_number
    sigma_h = sprm.sigma_holes
    hu = np.array([])
    hv = np.array([])
    
    # generate holes in the coverage
    if holes:
        print('Generate ' + str(Nh) + ' holes in the sampling pattern')
        while len(hu) < Nh:
            uv = -np.pi + 2 * np.pi * np.random.rand(2,1)
            if norm.pdf(0,0,sigma_h) * np.random.rand(1) > norm.pdf(np.linalg.norm(uv),0,sigma_h):
                hu = np.append(hu, uv[0])
                hv = np.append(hv, uv[1])
        
    # generate points outside the holes
    sigma_m = sprm.sigma
    Nm = int(sprm.p * sprm.N)
    hs = sprm.hole_size
    print('Generate ' + str(Nm) + ' frequency points')
    
    u = np.array([])
    v = np.array([])           
       
    while len(u) < Nm:
        Nmextra = int(Nm - len(u))
        if pattern == 'gaussian':
            us = sigma_m * np.random.randn(Nmextra)
            vs = sigma_m * np.random.randn(Nmextra)
        elif pattern == 'uniform':
            us = -np.pi + 2 * np.pi * np.random.rand(Nmextra)
            vs = -np.pi + 2 * np.pi * np.random.rand(Nmextra)
        elif pattern == 'geometric':
            us = np.random.geometric(sigma_m, Nmextra)
            vs = np.random.geometric(sigma_m, Nmextra)
        # discard points outside (-pi,pi)x(-pi,pi)
        sf1 = np.where((us < np.pi) & (us > -np.pi))[0]
        sf2 = np.where((vs < np.pi) & (vs > -np.pi))[0]
        sf = np.array(list(set(sf1).intersection(sf2)))
        
        if holes:
            for k in np.arange(Nh):
                # discard points inside the holes
                sf1 = np.where((u < hu[k] + hs) & (u > hu[k] - hs))[0]
                sf2 = np.where((v < hu[k] + hs) & (v > hu[k] - hs))[0]
                sfh = np.array(list(set(sf1).intersection(sf2)))
                sf = np.array(list(set(sf) - set(sfh)))
            
        if np.size(sf) > 0: 
            u = np.append(u, us[sf])
            v = np.append(v, vs[sf])
         
    print('Sampling pattern is done!')
    return u,v 

def util_gen_input_data(im, G, A, input_snr):
    '''
    Generate simulated data
    
    @param im: reference vectorized image
    
    @param G: function handle of convolutional kernel
    
    @param A: function handle of the direct operator
    
    @param input_snr: SNR in dB
    
    @return: (y, yn)Ideal observation of the reference image (using non-uniform FFT), Noisy observation
    '''
    
    import scipy.linalg as LA
    
    Av = A(im)              # Av = F * Z * Dr * Im
    y = G(Av.reshape(np.size(Av),1))            # y = G * Av
    N = np.size(y)
    
    sigma_noise = 10**(-input_snr/20.) * LA.norm(y)/np.sqrt(N)
    n = (np.random.randn(N,1) + 1j * np.random.randn(N,1)) * sigma_noise / np.sqrt(2) 
    
    yn = y + n
    
    return (y, yn)
    
def util_gen_l2_bounds(y, input_snr, param):
    '''
    Generate l2 ball bound and l2 ball stopping criterion
    
    @param y: input data, complex vector
    
    @param input_snr: signal to noise ratio of the input, defined by snr(dB) = 20 * log10( sqrt(mean (s^2)) / std(noise) )
    
    @param param: object of l2 ball parameters, more details to see the class "l2param"
    
    @return: l2 ball bound, l2 ball stopping criterion
     
    '''
    
    import scipy.linalg as LA
    from scipy.stats import chi2
    
    normy = LA.norm(y)
    N = np.size(y)
    sigma_noise = 10**(-input_snr/20.) * normy/np.sqrt(N)
    
    if param.l2_ball_definition == 'value' or param.l2_ball_definition == 1:
        epsilon = LA.norm(np.sqrt(2*np.size(y)))
        
    if param.l2_ball_definition == 'sigma' or param.l2_ball_definition == 2:
        s1 = param.bound        
        epsilon = np.sqrt(N + s1*np.sqrt(2*N)) * sigma_noise
    
    if param.l2_ball_definition == 'chi-percentile' or param.l2_ball_definition == 3:
        p1 = param.bound
        # isf(p, N) is inverse of survival function of chi2 distribution with N degrees of freedom 
        # to compute the minimum x so that the probability is no more than p
        epsilon = np.sqrt(chi2.isf(1-p1, N)) * sigma_noise 
        
    if param.stopping_criterion == 'l2-ball-percentage' or param.stopping_criterion == 1:
        epsilons = epsilon * param.stop
    
    if param.stopping_criterion == 'sigma' or param.stopping_criterion == 2:
        s2 = param.stop
        epsilons = np.sqrt(N + s2*np.sqrt(2*N)) * sigma_noise
        
    if param.stopping_criterion == 'chi-percentile' or param.stopping_criterion == 3:
        p2 = param.stop
        epsilons = np.sqrt(chi2.isf(1-p2, N)) * sigma_noise         
        
    return epsilon, epsilons
        

def operatorA(im, st, imsize, paddingsize):
    '''
    This function implements the operator A = F * Z * Dr
    
    @param im: input with imsize
    
    @param st: structure of nufft
    
    @param imsize: tuple of the image size
    
    @param paddingsize: tuple of the padding size
    
    @return: output with the padding size
    '''
    
    im = im.reshape(imsize)
    im = st.sn * im.astype('complex')           # Scaling (Dr)
    IM = scifft.fft2(im, paddingsize)                     # Oversampled Fourier transform
    return IM

def operatorAt(IMC, st, imsize, paddingsize):
    '''
    This function implements the adjoint operator At = Dr'Z'F'
    
    @param IMC: input with the padding size
    
    @param st: structure of nufft
    
    @param imsize: tuple of the image size
    
    @param paddingsize: tuple of the oversampled image size
    
    @return: output with the image size
    '''    
    
    imc = scifft.ifft2(IMC.reshape(paddingsize))      # F'
    im = st.sn.conj() * imc[:imsize[0],:imsize[1]]         # Dr'Z'F'
    return im

def operatorPhi(im, G, A, M=None):
    '''
    This function implements the operator Phi = G * A
    
    @param im: input image
    
    @param G: function handle of convolution kernel
    
    @param A: function handle of direct operator F * Z
    
    @param M: mask of the values that have no contribution to the convolution
    
    @return: visibilities, column complex vector
    '''
    spec = A(im)
    spec = np.reshape(spec,(np.size(spec),1))
    if M is not None:
        spec = spec[M,0]        
    vis = G(spec)
    return vis

def operatorPhit(vis, Gt, At, paddingsize=None, M=None):
    '''
    This function implements the operator Phi*T = A^T * G^T
    
    @param vis: input visibilities
    
    @param Gt: function handle of adjoint convolution kernel
    
    @param A: function handle of adjoint operator Z^T * F^T
    
    @param paddingsize: tuple of the oversampled image size, mandatory if M is not None
    
    @param M: mask of the values that have no contribution to the convolution
    
    @return: image, real matrix
    '''
    protospec = Gt(vis)
    if M is not None:
        protospec1 = np.zeros((paddingsize[0]*paddingsize[1],1)).astype('complex')
        protospec1[M] = protospec
        protospec = protospec1
    im = np.real(At(protospec))
    return im    

def operators(st):
    kernel = st.sp          
    imsize = st.Nd
    Kd = st.Kd
    
#     mask_G = np.any(kernel.toarray(),axis=0)        # mask of the values that have no contribution to the convolution 
#     kernel_m = kernel[:,mask_G]                   # convolution kernel after masking non-contributing values 
    
    mask_G = np.array((np.sum(np.abs(kernel.tocsc()), axis=0))).squeeze().astype('bool')        # faster computation and more economic storage
    kernel_m = kernel.tocsc()[:,mask_G]
    
    np.abs(kernel).sign()
    
    A = lambda x: operatorA(x, st, imsize, Kd)          # direct operator: F * Z * Dr
    At = lambda x: operatorAt(x, st, imsize, Kd)           # adjoint operator: Dr^T * Z^T * F^T
    
    G = lambda x: kernel.tocsr().dot(x.reshape(np.size(x),1))                      # convolutional kernel
    Gt = lambda x: kernel.tocsr().conj().T.dot(x.reshape(np.size(x),1))
    
    Gm = lambda x: kernel_m.tocsr().dot(x.reshape(np.size(x),1))                      # masked convolutional kernel
    Gmt = lambda x: kernel_m.tocsr().conj().T.dot(x.reshape(np.size(x),1))
    
    Phi = lambda x: operatorPhi(x, G, A)                # measurement operator: Phi = G * A
    Phi_t = lambda x: operatorPhit(x, Gt, At)
    
    Phim = lambda x: operatorPhi(x, Gm, A, mask_G)         # masked measurement operator: Phim = Gm * A
    Phim_t = lambda x: operatorPhit(x, Gmt, At, Kd, mask_G)
    
    return A, At, G, Gt, Gm, Gmt, Phi, Phi_t, Phim, Phim_t, mask_G

def guessmatrix(operator, M, N, diagonly=True):
    '''
    Compute the covariance matrix by applying a given operator (F*Phi^T*Phi) on different delta functions
    '''
    from scipy.sparse import coo_matrix
    from scipy.sparse import csc_matrix
    
    #Nx, Ny = imsize
    if diagonly:
        maxnonzeros = min(M, N)
        operdiag = np.zeros(maxnonzeros, dtype='complex')
    else:
#         matrix = np.zeros((M, N))               # HUGE
        matrix = csc_matrix((M, N))             # economic storage
        
    for i in np.arange(N):
        deltacol = coo_matrix(([1],([i],[0])),shape=(N,1))
        currcol = operator(deltacol.toarray()).flatten()
        if diagonly:
            if i > maxnonzeros: break
            operdiag[i] = currcol[i]
        else:
            matrix[:,i] = currcol[:,np.newaxis]
            
    if diagonly:
        matrix = coo_matrix((operdiag,(np.arange(maxnonzeros), np.arange(maxnonzeros))),shape=(M,N))
    
    return matrix

