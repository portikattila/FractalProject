import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numba import jit
import operator 
from scipy import stats

#X#X#X#X#X#X#X#X Part I.: Functions required to determine the fractal.

# Time evolution function -- describes the time evolution of the system
@jit(nopython=True) # I use numba to speed up the function
def MAP(u,v,w, p=1, n=3): 
    """ Time evolution of the quantum state.
        Paramenters: 
        - u,v,w :  the Bloch coordinates of the initial states
        - p : the complex parameter of the map 
        - n : the degree of the map 
        Returns: 
        - U,V,W : the Bloch coordinates of the final states"""
    
    n = int(n)
    p_sq = p*p
    p_abssq = np.abs(p)*np.abs(p)
    a = (1+w)**n
    b = (1-w)**n
    c = (u+1j*v)**n
    d = (u-1j*v)**n
    
    # the new coordinates
    N=(1+p_abssq)*((a+b))
    U=2*np.real((1-p_sq)*c+p*(b-a))/N
    V=2*np.imag((1+p_sq)*c+p*(a-b))/N
    W=(4*np.real(np.conjugate(p)*d)+(1-p_abssq)*(a-b))/N
    
    return U,V,W


#generate random points 
@jit(nopython=True, parallel=True) 
def random_points_on_sphere(n, r):
    """Random innitial states on surface of the Bloch sphere
        Parameters:
        - n : the number of states
        - r : the radius of the sphere
        Returns:
        - x,y,z : the coordinates of the states """
    n = int(n)
    u = np.random.uniform(-1, 1, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    
    x = r * (1 - u**2)**0.5 * np.cos(theta) 
    y = r * (1 - u**2)**0.5 * np.sin(theta)
    z = r * u 
    
    return x,y,z

@jit(nopython=True, parallel=True)
def random_points_in_ball(n, cx=0, cy=0, cz=0, r=1):
    """ Generate random points inside a sphere. 
        Paramenters: 
        - n : number of points 
        - r : the radiues of the ball
        - cx,cy,cz : the coordiantes of the center of the ball
        Returns: 
        - u,v,w - the Bloch coordinates of the random points """
    
    n = int(n)
    l=0
    u,v,w=np.zeros(n),np.zeros(n),np.zeros(n)
    while l<n:
        x,y,z=np.random.uniform(-r, r)+cx,np.random.uniform(-r, r)+cy,np.random.uniform(-r, r)+cz
        if ((x-cx)**2+(y-cy)**2+(z-cz)**2)<r**2:
            u[l],v[l],w[l] = x,y,z
            l+=1
    return u,v,w

@jit(nopython=True)
def _round(number, decimals):
    """ round the *number* to *decimals* decimal places (required for jit)"""
    return (number*10**decimals+0.5)//1/10**decimals

@jit(nopython=True, parallel=True)
def unique(u,v,w, decimals=6):
    """Calculate the unique values in the input arrays of the Bloch coordinates u,v,w."""
    U=[]
    V=[]
    W=[]
    x,y,z=_round(u,decimals),_round(v,decimals),_round(w,decimals)
    for i in np.unique(x):
        for j in np.unique(y[x==i]):
            for k in np.unique(z[np.logical_and(x==i, y==j)]):
                U.append(i)
                V.append(j)
                W.append(k)

    return np.array(U),np.array(V),np.array(W)

@jit(nopython=True)
def calculate_the_attractive_cycles(p,deg, number_of_states=1e4, number_of_iterations=500,r=0.99):
    """ Calculate the attractive cycles for given p (complex parameter) and deg (order of the protocol).
        Parameters:
        - p : the parameter of the map
        - deg : the degree of the map
        - number_of_states : number of initial states
        - number_of_iterations : the number of iterations
        Returns:
        U,V,W : the coordinates of the attractive sates
    """
    U,V,W=random_points_in_ball(number_of_states, r=r)#random initial states
    l=0
    k=0
    while(l!=len(U) and k<=10):
        k=k+1
        for i in range(number_of_iterations):
            U,V,W=MAP(U,V,W,p,deg) #evolve the states
        U,V,W=unique(U,V,W)#unique states
        l=len(U)
        for i in range(50): #check if the number of unique states does not change
            U,V,W=MAP(U,V,W,p,deg)
        U,V,W=unique(U,V,W)
        
    if k==10:
        print("More iteration needed.")
        return None
    return U,V,W

@jit(nopython=True, parallel=True)
def inverse_stereographic_projection(x, y, R = 1):
    """
    Inverse stereographic projection from the 2D plane to the surface of a sphere.
    Parameters:
    - x,y : the coordinates of the points 
    - R : the radius of the sphere
    Returns:
    - U,V,W : - 3D coordinates """
    
    temp = x**2 + y**2 + R**2
    
    u = 2 * x * R**2 / temp
    v = 2 * y * R**2 / temp
    w = (x**2 + y**2 - R**2) * R / temp

    r = np.sqrt(u**2 + v**2 + w**2)
    phi = np.arctan2(v, u)
    theta = np.arccos(w / r)

    U = r * np.sin(theta) * np.cos(phi)
    V = r * np.sin(theta) * np.sin(phi)
    W = r * np.cos(theta)
    return U,V,W

@jit(nopython=True)
def FS_2D(n,p,deg,P,xlow=-2,xhigh=2,ylow=-2,yhigh=2,epsilon=1e-3, number_of_iteration=100):
    """ Calculate the elements of the 2D Fractal set represted as the nodes of a grid. 
        Parameters: 
        - n : resolution, the dimension of the output array
        - p,deg : the parameter and the degree of the map
        - P : initial Purity of the states
        - xlow,xhigh,ylow,yhigh : the edges of the region
        - epsilon : numerical precision
        - number_of_iteration : number of iteration in one cycle
        Returns:
        - the elements of the fractal represented on a grid (nXn numpy array)"""
    #If there are two different values in one 2X2 subarray, we consider it as a point of the border between convergence regions 
    #i.e. the element of the fractal.
    
    box_size = 2 
    n = box_size * n
    R=np.zeros(n*n)
    
    #determine the attractive cycles
    Uf,Vf,Wf = calculate_the_attractive_cycles(p,deg,10000)
    
    #generate initial states uniformly
    x_grid=np.linspace(xlow,xhigh,n) #grid in real space
    y_grid=np.linspace(ylow,yhigh,n)
    R=np.zeros(n*n)
    x=np.zeros(n*n)
    y=np.zeros(n*n)
    for i in np.arange(n):
        for j in np.arange(n):
            x[i*n+j]=x_grid[i]
            y[i*n+j]=y_grid[j]
       
    #map the sates onto the Bloch sphere
    
    u,v,w=inverse_stereographic_projection(x,y,P)
    
    U=u
    #determine the converge regions
    # by iterating the function 
    while len(U)>len(Uf):
        for i in range(number_of_iteration):
            u,v,w=MAP(u,v,w,p,deg)
        U,V,W=unique(u,v,w)
    
    for k in range(len(Uf)):
        mask=(np.abs(u-Uf[k])<epsilon)
        mask=np.logical_and(mask, (np.abs(v-Vf[k])<epsilon))
        mask=np.logical_and(mask, (np.abs(w-Wf[k])<epsilon))
        if mask.any():
            R[mask]=k
    R=R.reshape((n,n))
    
    #determine the fractal set based on convergence regions
    
    number_of_boxes=int(n/box_size)
    x_grid=np.arange(0,n+box_size, box_size) #grid in representation space
    y_grid=np.arange(0,n+box_size, box_size)
    grid = np.zeros((int(n/box_size),int(n/box_size))) #grid array for storing results


    # iterate over the grid points (i,j) 
    for i in range(number_of_boxes):
        for j in range(number_of_boxes):
            # select the region "covered" by one grid point 
            T=R[x_grid[i]:x_grid[i+1],y_grid[j]:y_grid[j+1]]
            if np.unique(T).size > 1:
                grid[i,j] = 1

    return grid




@jit(nopython=True)
def norm(u,v,w):
    """ Calculate the NORM of the (u,v,w) point."""
    return np.sqrt(u**2 + v**2 + w**2) 

@jit(nopython=True)
def purity(u,v,w):
    """ Calculate the purity of state corespondin to the (u,v,w) point"""
    return 0.5 *(1 + NORM(u,v,w)**2)

@jit(nopython=True)
def nth_root(number, n,  k = 0):
    """ calculate the *k*th *n*-root of the  complex *number*"""
    r = np.abs(number)**(1/n)
    a = np.real(number)
    b = np.imag(number)
    phi = np.arctan2(b, a)
    return r * np.exp((1j * phi / n)) * np.exp(2j * np.pi / n)**k


def numerical_multiplier(n, p, z):
    """ Calculate the multiplier of the MAP_{*n,p*} at the *z* """
    return abs((n * (1 + abs(p)**2) * z**(n - 1)) / (1 - np.conjugate(p) * z**n )**2)

def calculate_of_repelling_fixed_points(n, p):
    """ Calculate the repelling pure fix points of the MAP_{*n,p*} """
    polynomial_coefficients = np.zeros(n + 2, dtype = complex)
    polynomial_coefficients[-2] = -1
    polynomial_coefficients[0] = np.conjugate(p)
    polynomial_coefficients[1] = 1
    polynomial_coefficients[-1] = p
    roots = np.roots(polynomial_coefficients)
    rep = roots[numerical_multiplier(n, p, roots) > 1]
    return rep

@jit(nopython=True, parallel=True)
def inverse_iteration(deg, number_of_iteration, p, z0):
    """ Calculate the Julia set via backiteration, i.e. the iterration of the inverse MAP
        Parameters:
        - deg : the oreder of the map
        - number_of_iteration - number of iteration, equal to the number of generated fractal points
        - p : parameter of the map
        - z0 : starting point of the iteration
        Returns: 
        - x,y : the coordinates of the points """
    
    deg = int(deg)
    N = int(number_of_iteration)
    
    x = np.empty(N)
    y = np.empty(N)
    
    x[0] = np.real(z0)
    y[0] = np.imag(z0)
    
    z = z0
    for i in np.arange(1, N):
        k = np.floor(np.random.uniform(0, deg))
        z = nth_root(((p - z) / (1 + np.conjugate(p)*z)), deg, k)
        x[i] = np.real(z)
        y[i] = np.imag(z)
    
    return x,y

def FS_backiteration(n,p,deg,xlow=-2,xhigh=2,ylow=-2,yhigh=2, number_of_iteration=1000):
    """ Calculate the elements of the Fractal set via backiteration 
        Parameters: 
        - n : grid resolution, the dimension of the output array
        - p, deg : the parameter and the degree of the map
        - xlow,xhigh,ylow,yhigh : the edges of the region
        - number_of_iteration : number of iteration
        Returns:
        - the elements of the fractal represented on a grid (nXx numpy array)"""

    z0 = calculate_of_repelling_fixed_points(2,1)
    x = np.array([])
    y = np.array([])
    for i in z0:
        x_temp,y_temp = inverse_iteration(deg,number_of_iteration,p,i)
        x = np.concatenate((x,x_temp))
        y = np.concatenate((y,y_temp))   

    R = box_counter((x,y),bins=n,r=[[xlow,xhigh],[ylow,yhigh]],fig=True)   
    

    return R



def FS_3D(b,n,p,deg, P=1, epsilon=1e-3, number_of_iteration=100):
    """ Calculate the elements of the 3D Fractal set on grid 
        Parameters: 
        - b : resulution of the output
        - n :  number of (randomly choosed) initial points
        - p, deg : the parameter and the degree of the map
        - P : initial Purity
        - xlow,xhigh,ylow,yhigh : the edges of the region
        - epsilon : numerical precision
        - number_of_iteration : number of iteration in one cycle
        Returns:
        - the elements of the fractal represented on a grid (nXx numpy array)"""
 
    Uf,Vf,Wf = calculate_the_attractive_cycles(p,deg,10000)
    u,v,w = random_points_in_ball(n,r=np.sqrt(2*P-1))
    u0,v0,w0 = u,v,w

    R=np.zeros(n)
    

    U=u
    #determine the converge regions
    while len(U)>len(Uf):
        for i in range(number_of_iteration):
            u,v,w=MAP(u,v,w,p,deg)
        U,V,W=unique(u,v,w)
    coordinates = []

    for k in range(len(Uf)):
        mask=(np.abs(u-Uf[k])<epsilon)
        mask=np.logical_and(mask, (np.abs(v-Vf[k])<epsilon))
        mask=np.logical_and(mask, (np.abs(w-Wf[k])<epsilon))
        if mask.any():
            coordinates.append((u0[mask],v0[mask],w0[mask]))

    T = np.zeros((b,b,b))
    for k in range(len(Uf)):
        T = T + box_counter_3D(coordinates[k], bins=b)
    
    u,v,w = np.where(T>1)
    
    u = 2*u/b-1
    v = 2*v/b-1
    w = 2*w/b-1
    
    R = np.zeros_like(T)
    R[T>1] = 1
    return R, u,v,w


#X#X#X#X#X#X#X#X Part II.: Functions required to determine the Box-counting dimension

@jit(nopython=True)
def box_number_2D(box_sizes,n,p,deg,P,xlow=-2,xhigh=2,ylow=-2,yhigh=2,epsilon=1e-3, number_of_iteration=100):
    """ Calculate the number of boxes required to cover the Fractal set for given box sizes. 
        Parameters: 
        - box_sizes : size of boxes 
        - n : sample resolution
        - p, deg : the parameter and the degree of the map
        - P : initial Purity
        - xlow,xhigh,ylow,yhigh : the edges of the region
        - epsilon : numerical precision
        - number_of_iteration : number of iteration in one cycle
        Returns:
        - box_sizes, bn : the (box size, number of boxes) data pairs"""

    Uf,Vf,Wf = calculate_the_attractive_cycles(p,deg,10000)
    #generate initial states uniformly
    x_grid=np.linspace(xlow,xhigh,n) #grid in real space
    y_grid=np.linspace(ylow,yhigh,n)
    R=np.zeros(n*n)
    x=np.zeros(n*n)
    y=np.zeros(n*n)
    for i in np.arange(n):
        for j in np.arange(n):
            x[i*n+j]=x_grid[i]
            y[i*n+j]=y_grid[j]
       
    
    u,v,w=inverse_stereographic_projection(x,y,P)
    
    U=u
    #determine the converge regions
    while len(U)>len(Uf):
        for i in range(number_of_iteration):
            u,v,w=MAP(u,v,w,p,deg)
        U,V,W=unique(u,v,w)
    
    for k in range(len(Uf)):
        mask=(np.abs(u-Uf[k])<epsilon)
        mask=np.logical_and(mask, (np.abs(v-Vf[k])<epsilon))
        mask=np.logical_and(mask, (np.abs(w-Wf[k])<epsilon))
        if mask.any():
            R[mask]=k
    R=R.reshape((n,n))
     
    bn = np.zeros(box_sizes.size)
        
    for b,box_size in enumerate(box_sizes):
        
        number_of_boxes=int(n/box_size)

        x_grid=np.arange(0,n+box_size, box_size) #grid in representation space
        y_grid=np.arange(0,n+box_size, box_size)
        grid = np.zeros((int(n/box_size),int(n/box_size))) #grid array for storing results


        # iterate over the grid points (i,j)
        for i in range(number_of_boxes):
            for j in range(number_of_boxes):
                # select the region "covered" by one grid point 
                T=R[x_grid[i]:x_grid[i+1],y_grid[j]:y_grid[j+1]]
                if np.unique(T).size > 1:
                    grid[i,j] = 1
        bn[b] = np.sum(grid)
   
    return box_sizes, bn, R


def box_counter(I, bins = 10, r = None, hist = None, fig = None):
    """ Calculate the number of boxes required to cover a set of points.
        Parameters:
        - I : points of set - (N, M) array or sequence, -np.array([x, y]) or (x,y) 
        - bins - number of bins - sequence or int                                               
        - r - the range of space which contains the set                 
            - sequence of length M                                      
        - hist - if hist is True, the result is a histogram                                            
        - fig - if fig is True, results contain the figure about set                                                                
     
       Returns:                                                        
       - number_of_boxes - the number of boxes needed to cover the set 
       - h - |if hist is True| - histomgram of the set                 
       - b - figure about the set as an array of shape (N, M)"""
    
    # checking the parameters
    
    try: #I is an N x M sheped array, N - lenght of time series, M - the dimension of the set
        N, M = I.shape  
    
    except (AttributeError, ValueError): #I is a sequence(tuple) of 1D arrays
        I = np.atleast_2d(I).T # convert to numpy array
        N, M = I.shape

    nbin = np.empty(M, int) # number of bins
    edges = M * [None] # edges between the bins
    dedges = M * [None] # width of bins
    
    try:
        n = len(bins)
        if n != N:
            raise ValueError( 'The dimension of bins must be equal to the dimension of the input array.')
    except TypeError: # if bins is an integer ---> number of bins
        bins = M*[bins]

    if r is None: # if the range is not specified
        r = (None,) * M
    elif len(r) != M: # the range should be given on every axis
        raise ValueError('r argument must have one entry per dimension.')

    #calculation the edges
    
    for i in range(M): # calculate the edges
        if np.ndim(bins[i]) == 0: # checking if the number is an integer
            if bins[i] < 1: 
                raise ValueError('The elements of the bins must be positive integers.')
            if r[i] is not None:
                smin, smax = r[i]
            else:
                smin, smax = np.min(I[:,i]), np.max(I[:,i])
            if smin == smax: # if range is empty
                smin = smax - 0.5
                smax = smax + 0.5
                
            try:
                n = operator.index(bins[i])
            except:
                raise TypeError('The elements of the bins must be positive integers.')
            edges[i] = np.linspace(smin, smax, n + 1)
        else:
            raise ValueError('bins must be an integer or 1d array of integers.')
    
        nbin[i] = len(edges[i]) + 1 
        dedges[i] = np.diff(edges[i])


    # calculation bin number
    
    #find the indexes of the bin which contains the value
    #values that fall on an edge are put in the right bin
    Ncount = tuple(np.searchsorted(edges[i], I[:, i], side = 'right') for i in range(M)) 
    
    #values that fall on the last edge should be counted in the last bin
    for i in range(M):
        on_edge = (I[:, i] == edges[i][-1]) # points on the last edge
        Ncount[i][on_edge] -= 1 #shift to the previous bin on left
    
    xy = np.ravel_multi_index(Ncount, nbin) # convertne idexes to flat representaion
    
    
    b = np.bincount(xy, minlength = nbin.prod()) # occurrences of indexes in array -> histogram
    b = b.reshape(nbin) # reshape the array
    b = b.astype(float, casting = 'safe')

    # remove outliers
    core = M * (slice(1, -1),)
    b = b[core]
    if hist:
        return b
    b[b != 0] = 1 # set the value of the non-empty bin to one
    number_of_boxes = np.sum(b) # count the number of boxes
    
    if fig:
        return b
    
    return number_of_boxes

def box_counter_3D(I, bins = 10):
    """ Calculate the number of boxes required to cover a set of 3D points.
    Parameters:
    - I : points of set - (N, M,M) array or sequence, -np.array([x,y,z]) or (x,y,z) 
    - bins - number of bins - sequence or int                                               
    Returns:                                                        
    - b : the number of boxes needed to cover the set"""
      
    I = np.atleast_2d(I).T # convert to numpy array
    N, M = I.shape

    nbin = np.empty(M, int) # number of bins
    edges = M * [None] # edges between the bins
    dedges = M * [None] # width of bins
    bins = M*[bins]

    for i in range(M): # calculate the edges
        if np.ndim(bins[i]) == 0: # checking if the number is an integer
            if bins[i] < 1: 
                raise ValueError('The elements of the bins must be positive integers.')
            smin, smax = -1,1 # defining range
                
            n = operator.index(bins[i])
            edges[i] = np.linspace(smin, smax, n + 1)
        
        nbin[i] = len(edges[i]) + 1 
        dedges[i] = np.diff(edges[i])

    #find the indexes of the bin which contains the value
    #values that fall on an edge are put in the right bin
    Ncount = tuple(np.searchsorted(edges[i], I[:, i], side = 'right') for i in range(M)) 
    
    #values that fall on the last edge should be counted in the last bin
    for i in range(M):
        on_edge = (I[:, i] == edges[i][-1]) # points on the last edge
        Ncount[i][on_edge] -= 1 #shift to the previous bin on left
    
    xy = np.ravel_multi_index(Ncount, nbin) # convertne idexes to flat representaion
    
    
    b = np.bincount(xy, minlength = nbin.prod()) # occurrences of indexes in array -> histogram
    b = b.reshape(nbin) # reshape the array
    b = b.astype(float, casting = 'safe')

    # remove outliers
    core = M * (slice(1, -1),)
    b = b[core]
    b[b != 0] = 1 # set the value of the non-empty bin to one
    
    return b


def box_number_back_iteration(box_sizes,n,p,deg,xlow=-2,xhigh=2,ylow=-2,yhigh=2, number_of_iteration=1000):
    """ Calculate the number of boxes required to cover the points of a Fractal set for given box sizes. 
        Parameters: 
        - box_sizes : size of boxes 
        - n : grid resolution
        - p, deg : the parameter and the degree of the map
        - P : initial Purity
        - xlow,xhigh,ylow,yhigh : the edges of the region
        - number_of_iteration : number of iteration in one cycle
        Returns:
        - box_sizes, bn : the (box size, number of boxes) data pairs"""
   
    z0 = calculate_of_repelling_fixed_points(2,1)
    x = np.array([])
    y = np.array([])
    for i in z0:
        x_temp,y_temp = inverse_iteration(deg,number_of_iteration,p,i)
        x = np.concatenate((x,x_temp))
        y = np.concatenate((y,y_temp))   

    R = box_counter((x,y),bins=n,r=[[xlow,xhigh],[ylow,yhigh]],fig=True)   
    R[R>0]=1



    bn = np.zeros(box_sizes.size)
        
    for b,box_size in enumerate(box_sizes):
        
        number_of_boxes=int(n/box_size)

        x_grid=np.arange(0,n+box_size, box_size) #grid in representation space
        y_grid=np.arange(0,n+box_size, box_size)
        grid = np.zeros((int(n/box_size),int(n/box_size))) #grid array for storing results


        # iterate over the grid points (i,j)
        for i in range(number_of_boxes):
            for j in range(number_of_boxes):
                # select the region "covered" by one grid point 
                T=R[x_grid[i]:x_grid[i+1],y_grid[j]:y_grid[j+1]]
                if np.unique(T).size > 1:
                    grid[i,j] = 1
        bn[b] = np.sum(grid)
   
    return box_sizes, bn, R

def box_counting_dimension_2D(box_sizes,n,p,deg,P,xlow=-2,xhigh=2,ylow=-2,yhigh=2,epsilon=1e-3, number_of_iteration=10000000,method="iteration", i=0,j=-1,plot=False):
    """ Calculates the Box-counting Dimension 
    Parameters: 
    - box_sizes : size of boxes 
    - n : sample resolution
    - p, deg : the parameter and the degree of the map
    - P : initial Purity
    - xlow,xhigh,ylow,yhigh : the edges of the region
    - epsilon : numerical precision
    - number_of_iteration : number of iteration in one cycl
    - i,j : imput array first and last index
    Returns:
    - box_sizes, bn : the (box size, number of boxes) data pairs """
    
    if method == "iteration":
        s,bn, R = box_number_2D(box_sizes,n,p,deg,P,xlow,xhigh,ylow,yhigh,epsilon, number_of_iteration)
        R = np.abs(np.diff(R, axis=0)[:,:n-1] + np.diff(R, axis=1)[:n-1,:])
        R[R>0]=1
    elif method == "backiteration":
        s,bn, R = box_number_back_iteration(box_sizes,n,p,deg,xlow,xhigh,ylow,yhigh, number_of_iteration=n*n)
    
    result = stats.linregress(np.log(s[i:j]), np.log(bn[i:j]))
    
    if plot:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4), gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle('Box-counting dimension: degree = {}, p = {}'.format(deg, p), fontsize=20)
        ax1.plot(np.log(s[i:j]), np.log(bn[i:j]),'r.--')
        ax1.plot(np.log(s[i:j]), result.slope*np.log(s[i:j])+result.intercept)
        ax1.grid()
        
        ax2.imshow(R, cmap="binary")
        plt.show()
        
        print("Box-counting Dimension:", np.round(-result.slope,3),'+/-',np.round(result.stderr,3))
        
    return np.round(-result.slope,3)

def box_number_3D(bins,n,p,deg, P=1, epsilon=1e-3, number_of_iteration=100):
    """ Calculate the number of boxes required to cover the 3D Fractal set for given box sizes. 
        Parameters: 
        - bins : number of boxes 
        - n : number of initial states
        - p, deg : the parameter and the degree of the map
        - P : initial Purity
        - xlow,xhigh,ylow,yhigh : the edges of the region
        - epsilon : numerical precision
        - number_of_iteration : number of iteration in one cycle
        Returns:
        - box_sizes, bn : the (box size, number of boxes) data pairs"""
 
    Uf,Vf,Wf = calculate_the_attractive_cycles(p,deg,10000)
    u,v,w = random_points_in_ball(n,r=np.sqrt(2*P-1))
    u0,v0,w0 = u,v,w

    R=np.zeros(n)
    u,v,w = random_points_in_ball(n,r=np.sqrt(2*P-1))

    U=u
    #determine the converge regions
    while len(U)>len(Uf):
        for i in range(number_of_iteration):
            u,v,w=MAP(u,v,w,p,deg)
        U,V,W=unique(u,v,w)
    coordinates = []

    for k in range(len(Uf)):
        mask=(np.abs(u-Uf[k])<epsilon)
        mask=np.logical_and(mask, (np.abs(v-Vf[k])<epsilon))
        mask=np.logical_and(mask, (np.abs(w-Wf[k])<epsilon))
        if mask.any():
            coordinates.append((u0[mask],v0[mask],w0[mask]))

    number_of_boxies = []
    for b in bins:
        T = np.zeros((b,b,b))
        for k in range(len(Uf)):
            T = T + box_counter_3D(coordinates[k], bins=b)
        number_of_boxies.append(np.sum(T>1))
        
    return bins,number_of_boxies

def box_counting_dimension_3D(bins,n,p,deg,P=1,epsilon=1e-3, number_of_iteration=1000, i=0,j=-1,plot=False):
    """ Calculates the Box-counting Dimension 
    Parameters: 
    - bins : number of boxes 
    - n : number of initial states
    - p, deg : the parameter and the degree of the map
    - P : initial Purity
    - epsilon : numerical precision
    - number_of_iteration : number of iteration in one cycl
    - i,j : imput array first and last index
    Returns:
    - box_sizes, bn : the (box size, number of boxes) data pairs """
    
    s,bn = box_number_3D(bins,n,p,deg,P,epsilon, number_of_iteration)
    result = stats.linregress(np.log(s[i:j]), np.log(bn[i:j]))
    
    if plot:
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 

        # Set up a figure twice as tall as it is wide
        fig = plt.figure(figsize=(12,4))

        fig.suptitle('Box-counting dimension: degree = {}, p = {}'.format(deg, p), fontsize=20)


        # Second subplot
        ax = fig.add_subplot(gs[0])
        ax.plot(np.log(s[i:j]), np.log(bn[i:j]),'r.--')
        ax.plot(np.log(s[i:j]), result.slope*np.log(s[i:j])+result.intercept)
        ax.grid()

        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)

        x = np.outer(np.sin(u), np.sin(v))
        y = np.outer(np.sin(u), np.cos(v))
        z = np.outer(np.cos(u), np.ones_like(v))

        cx1 = np.sin(v)
        cy1 = np.cos(v) 
        cz1 = np.zeros_like(cy1)

        cx2 = np.sin(v)
        cy2 = np.zeros_like(cx2)
        cz2 = np.cos(v) 

        cx3 = np.zeros_like(cy1)
        cy3 = np.sin(v)
        cz3 = np.cos(v) 

        a1x = np.linspace(-1, 1, 5)
        a1y = np.zeros_like(a1x)
        a1z = np.zeros_like(a1x)

        a2x = np.zeros_like(a1x)
        a2y = np.linspace(-1, 1, 5)
        a2z = np.zeros_like(a1x)

        a3x = np.zeros_like(a1x)
        a3y = np.zeros_like(a1x)
        a3z = np.linspace(-1, 1, 5)

        ax = fig.add_subplot(gs[1], projection = '3d')

        ax.xaxis.set_pane_color((1.0, 1.0, 0.5, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax._axis3don = False

        ax.plot_wireframe(x, y, z, colors = 'gray', alpha = 0.1)

        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)
        x = np.outer(np.sin(u), np.sin(v))
        y = np.outer(np.sin(u), np.cos(v))
        z = np.outer(np.cos(u), np.ones_like(v))
        ax.plot_surface(0.99*x, 0.99*y, 0.99*z, color='pink', alpha=0.2, zorder=1)

        ax.plot(cx1, cy1, cz1, 'k-', alpha = 0.6, linewidth = 1., zorder=2)
        ax.plot(cx2, cy2, cz2, 'k-', alpha = 0.6, linewidth = 1., zorder=2)
        ax.plot(cx3, cy3, cz3, 'k-', alpha = 0.6, linewidth = 1., zorder=2)

        ax.plot(a1x, a1y, a1z, 'k--', alpha = 0.4, linewidth = 1., zorder=2)
        ax.plot(a2x, a2y, a2z, 'k--', alpha = 0.4, linewidth = 1., zorder=2)
        ax.plot(a3x, a3y, a3z, 'k--', alpha = 0.4, linewidth = 1., zorder=2)

        ax.text(0, 0, 1.2, '$ |0 \\rangle $', fontsize = '10')
        ax.text(0, 0, -1.4, '$ |1 \\rangle $', fontsize = '10')



        x,y = inverse_iteration(deg,100000,p,1)
        u,v,w = inverse_stereographic_projection(x,y,1)

        ax.plot(u,v,w, 'r,', zorder=3)

        ax.set_xlim(-0.65,0.65)
        ax.set_ylim(-0.65,0.65)
        ax.set_zlim(-0.5,0.5)


        plt.show()
        
        print("Box-counting Dimension:", np.round(result.slope,3),'+/-',np.round(result.stderr,3))
        
    return np.round(result.slope,3)

#X#X#X#X#X#X#X#X Part III.: Functions required to determine the Correlation Dimension


@jit(nopython=True)
def calculate_distance_matrix(points):
    """
    Calculate the distance matrix between all pairs of points.
    Parameters:
    - points : the coordinates of the points - array([{x_i}])
    Returns:
    - distance : distance matrix
    """
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            distances[i, j] = np.linalg.norm(points[i] - points[j])
            distances[j, i] = distances[i, j]  # Distance matrix is symmetric
    return distances

@jit(nopython=True)
def correlation_integral(distances, r_values):
    """
    Calculate the correlation integral for different distance thresholds (r values).
    Parameters:
    - distance : distance matrix
    - r_values : distance treshold values
    Returns:
    - correlations : correlations
    """
    num_points = len(distances)
    correlations = []
    for r in r_values:
        count = np.sum(distances <= r) - num_points  # Exclude diagonal elements
        correlations.append(count)
    return np.array(correlations)


def correlation_dimension_2D(p,deg, r_values, plot=False, i=0,j=-1, num_points=500, num_iteration=1000000):
    """
    Calculate the correlation dimension of a set of points.
    Parameters:
    - points : the coordinates of the points - array([{x_i}])
    - r_values : distance treshold values
    Returns:
    - correlation dimension 
    """
    z0 = calculate_of_repelling_fixed_points(deg, p)
    x,y = inverse_iteration(deg, num_iteration,p,z0[0])
    points = np.array([x,y]).T
    
    points = points[np.random.choice(range(len(x)), size=num_points)]

    
    distances = calculate_distance_matrix(points)
    correlations = correlation_integral(distances, r_values)
    C_r = correlations / len(points)
    
    # Remove r=0 to avoid division by zero
    nonzero_indices = np.nonzero(r_values)
    C_r = C_r[nonzero_indices]
    r_values = r_values[nonzero_indices]
    
    
    # Fit a linear regression to find the slope (correlation dimension)
    result = stats.linregress(np.log(r_values), np.log(C_r))
    
    # plot 
    if plot:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4), gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle('Correlation dimension: degree = {}, p = {}'.format(deg, p), fontsize=20)

        ax1.plot(np.log(r_values[i:j]), np.log(C_r[i:j]),'r.--')
        ax1.plot(np.log(r_values[i:j]), result.slope*np.log(r_values[i:j])+result.intercept)
        ax1.grid()

        ax2.plot(x,y, "k,")
        plt.show()
        print("Correlation Dimension:", np.round(-result.slope,3),'+/-',np.round(result.stderr,3))
    return result.slope

def correlation_dimension_3D(p,deg, r_values, plot=False, i=0,j=-1, num_points=500, num_iteration=1000000,):
    """
    Calculate the correlation dimension of a set of points.
    Parameters:
    - points : the coordinates of the points - array([{x_i}])
    - r_values : distance treshold values
    Returns:
    - correlation dimension 
    """
    _,u,v,w = FS_3D(num_points,num_iteration,p,deg)
    points = np.array([u,v,w]).T
    
    
    distances = calculate_distance_matrix(points)
    correlations = correlation_integral(distances, r_values)
    C_r = correlations / len(points)
    
    # Remove r=0 to avoid division by zero
    nonzero_indices = np.nonzero(r_values)
    C_r = C_r[nonzero_indices]
    r_values = r_values[nonzero_indices]
    
    
    # Fit a linear regression to find the slope (correlation dimension)
    result = stats.linregress(np.log(r_values), np.log(C_r))
    
    # plot 
    if plot:
        fig, ax = plt.subplots(figsize=(8,4))
        fig.suptitle('Correlation dimension: degree = {}, p = {}'.format(deg, p), fontsize=20)

        ax.plot(np.log(r_values[i:j]), np.log(C_r[i:j]),'r.--')
        ax.plot(np.log(r_values[i:j]), result.slope*np.log(r_values[i:j])+result.intercept)
        ax.grid()
        plt.show()

    print("Correlation Dimension:", np.round(result.slope,3),'+/-',np.round(result.stderr,3))
    
    return result.slope


#X#X#X#X#X#X#X#X Part IV.: Functions required to determine the Multifractal spectrum

def calculate_multifractal_spectrum_2D(I, box_sizes, Q):
    """ 
    Calculate the Multifractal spectrum with the box-counting methode, based on 10.1103/PhysRevLett.62.1327
    Parameters:
    - I : input array, grid representation of the fractal
    - box_sizes : array
    - Q : exponents for distorting, arbitary interval
    Returns :
    results : dict, contains: {Fractal, Q, alpha(Q), f(Q), D(Q), t(Q }   
    """
    
    Q = Q[np.abs(1-Q)>1e-3] # avoid Q=1 --> unique value, will handle it later
    
    # prepare array for results
    I_qe = np.zeros((box_sizes.size, Q.size))
    alpha_qe = np.zeros((box_sizes.size, Q.size))
    f_qe = np.zeros((box_sizes.size, Q.size))
    
    # 3 nested loop to calculate the quatities for every q and epsilon values
    for box_sizei, box_size in enumerate(box_sizes):
        rows, cols = I.shape
        num_boxes_per_row = rows // box_size
        num_boxes_per_col = cols // box_size
        m_i = [] #number of pixels in ith box --> mass distribution function

        # calculate the mass probability distribution function for different boxsizes
        for i in range(num_boxes_per_row):
            for j in range(num_boxes_per_col):
                sub_I = np.sum(I[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size])
                if sub_I > 0:
                    m_i.append(np.sum(sub_I))

        P = np.array(m_i)/np.sum(m_i)

        # determine the Q dependent quantities
        for qi, q in enumerate(Q):
            I_qe[box_sizei, qi]=np.sum(P**q)
            alpha_qe[box_sizei, qi] = np.sum(P**q*np.log(P))/np.sum(P**q) 
            f_qe[box_sizei, qi] = np.sum((P**q/np.sum(P**q))*np.log((P**q/np.sum(P**q))))

    # determine the averaged quantites by linear regression
    tau_q = np.zeros_like(Q)
    D_q = np.zeros_like(Q)    #generalized dimension
    alpha_q =np.zeros_like(Q) 
    f_q =np.zeros_like(Q)


    for qi,q in enumerate(Q):
        results = stats.linregress(np.log(box_sizes), np.log(I_qe[:,qi]))
        tau_q[qi] = results.slope

        results = stats.linregress(-np.log(box_sizes), np.log(I_qe[:,qi]))
        D_q[qi] = results.slope/(1-q)

        results = stats.linregress(np.log(box_sizes), alpha_qe[:,qi])
        alpha_q[qi] = results.slope

        results = stats.linregress(np.log(box_sizes), f_qe[:,qi])
        f_q[qi] = results.slope
        
        results = dict()
    
    results['Fractal'] = I
    results['Q'] = Q
    results['f(Q)'] = f_q
    results['alpha(Q)'] = alpha_q
    results['D(Q)'] = D_q
    results['t(Q)'] = tau_q
    
    return results

def calculate_multifractal_spectra_3D(I, box_sizes, Q):
    """ 
    Calculate the Multifractal spectrum with the box-counting methode, based on 10.1103/PhysRevLett.62.1327
    Parameters:
    - I : input array, grid representation of the fractal
    - box_sizes : array
    - Q : exponents for distorting, arbitary interval
    Returns :
    results : dict, contains: {Fractal, Q, alpha(Q), f(Q), D(Q), t(Q }   
    """
    
    Q = Q[np.abs(1-Q)>1e-3]

    I_qe = np.zeros((box_sizes.size, Q.size))
    alpha_qe = np.zeros((box_sizes.size, Q.size))
    f_qe = np.zeros((box_sizes.size, Q.size))

    for box_sizei, box_size in enumerate(box_sizes):
        x, y,z = I.shape
        num_boxes_per_x = x // box_size
        num_boxes_per_y = y // box_size
        num_boxes_per_z = z // box_size
        
        m_i = [] #number of pixels in ith box

        for i in range(num_boxes_per_x):
            for j in range(num_boxes_per_y):
                for k in range(num_boxes_per_z):
                    sub_I = np.sum(I[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size, k*box_size:(k+1)*box_size])
                    if sub_I > 0:
                        m_i.append(np.sum(sub_I))

        P = np.array(m_i)/np.sum(m_i)


        for qi, q in enumerate(Q):
            I_qe[box_sizei, qi]=np.sum(P**q)
            alpha_qe[box_sizei, qi] = np.sum(P**q*np.log(P))/np.sum(P**q) 
            f_qe[box_sizei, qi] = np.sum((P**q/np.sum(P**q))*np.log((P**q/np.sum(P**q))))

    #generalized dimension
    tau_q = np.zeros_like(Q)
    D_q = np.zeros_like(Q)
    alpha_q =np.zeros_like(Q)
    f_q =np.zeros_like(Q)


    for qi,q in enumerate(Q):
        results = stats.linregress(np.log(box_sizes), np.log(I_qe[:,qi]))
        tau_q[qi] = results.slope

        results = stats.linregress(-np.log(box_sizes), np.log(I_qe[:,qi]))
        D_q[qi] = results.slope/(1-q)

        results = stats.linregress(np.log(box_sizes), alpha_qe[:,qi])
        alpha_q[qi] = results.slope

        results = stats.linregress(np.log(box_sizes), f_qe[:,qi])
        f_q[qi] = results.slope
        
        results = dict()
        
    results['Q'] = Q
    results['f(Q)'] = f_q
    results['alpha(Q)'] = alpha_q
    results['D(Q)'] = D_q
    results['t(Q)'] = tau_q
    
    return results

def plot_multifractal_spectrum_2D(R, deg, p):
    """
    Simple function to plot the multifractal spectra from results
    """
    
    fig, ax = plt.subplots(1,2,figsize=(12,4), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle('Multifractal spectrum: degree = {}, p = {}'.format(deg, p), fontsize=20)
    ax[0].plot(R['alpha(Q)'], R['f(Q)'], '.--')
    ax[0].set_xlabel('$\\alpha$(Q)', fontsize=15)
    ax[0].set_ylabel('f(Q)', fontsize=15)
    ax[0].grid()
    
    
    ax[1].imshow(R['Fractal'], cmap="binary")
    ax[1].axis('off')
    plt.show()
    
    
    fig, ax = plt.subplots(1,2,figsize=(12,4), gridspec_kw={'width_ratios': [1, 1]})
    ax[0].plot(R['Q'], R['f(Q)'], '.--')

    ax[0].set_xlabel('Q', fontsize=15)
    ax[0].set_ylabel('f(Q)', fontsize=15)
    ax[0].grid()

    ax[1].plot(R['Q'], R['D(Q)'], '.--', label='D(Q)')
    ax[1].plot(R['Q'], R['alpha(Q)'], '.--', label='$\\alpha$(Q)')
    
    ax[1].set_xlabel('Q', fontsize=15)
    ax[1].legend()
    ax[1].grid()
    plt.show()

def plot_multifractal_spectrum_3D(R, deg, p):
    """
    Simple function to plot the multifractal spectra from results
    """
    
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    fig.suptitle('Multifractal spectrum: degree = {}, p = {}'.format(deg, p), fontsize=20)

    ax[0].plot(R['alpha(Q)'], R['f(Q)'], '.--')
    ax[0].set_xlabel('$\\alpha(Q)$', fontsize=15)
    ax[0].set_ylabel('f(Q)', fontsize=15)
    ax[0].grid()
    
    
    ax[1].plot(R['alpha(Q)'], R['t(Q)'], '.--')
    ax[1].set_xlabel('$\\alpha$(Q)', fontsize=15)
    ax[1].set_ylabel('$\\tau$(Q)', fontsize=15)
    ax[1].grid()
    plt.show()
    
    
    fig, ax = plt.subplots(1,2,figsize=(12,4), gridspec_kw={'width_ratios': [1, 1]})
    ax[0].plot(R['Q'], R['f(Q)'], '.--')

    ax[0].set_xlabel('Q', fontsize=15)
    ax[0].set_ylabel('f(Q)', fontsize=15)
    ax[0].grid()

    ax[1].plot(R['Q'], R['D(Q)'], '.--', label='D(Q)')
    ax[1].plot(R['Q'], R['alpha(Q)'], '.--', label='$\\alpha$(Q)')
    
    ax[1].set_xlabel('Q', fontsize=15)
    ax[1].legend()
    ax[1].grid()
    plt.show()