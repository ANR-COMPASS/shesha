import numpy as np

"""
---------------------------------------------------------------------------------------------------------
File name: slopesCovariance.py
Authors: E. Gendron & F. Vidal
Python Version: 3.x
Re-written in Python from the original code in Yorick made in 2010 for the CANARY experiment.

See also Vidal F., Gendron E. and Rousset G, "Tomography approach for multi-object adaptive optics
", JOSA-A Vol. 27, Issue 11, pp. A253-A264 (2010)

This file contains all the routines needed to compute slopes covariance matrices between 2 SH WFS and serves as 
a basis for the Learn & Apply algorithm.

The main function to be called should be: compute_Caa (see example in the function docstring)
compute_Caa is a low-level function. 

To be used properly, compute_Caa shall be called wrapped into several loops:
- 2 nested loops over all the WFSs of the experiment
- 1 other nested loop (with sum) over the layers
in order to fill the whole covariance matrix of the system and
integrate over the Cn2(h) profile.

The fitting process (LEARN part) shall use a minimization algorithm (such as https://lmfit.github.io/lmfit-py/model.html) 
to compute the minimum mean square error between the measured covariance slopes matrix from WFSs data 
and the slopes covariance matrix model provided by this file to estimate the desired atmospheric parameters 
(I.e fitted parameters are usually Cn2 parameters such as r_0 and altitude). 

A special care shall be taken in slopes order in the pupil (see example docstring in compute_Caa).

---------------------------------------------------------------------------------------------------------
"""



def KLmodes(x, y, L0, filterPiston):
    '''
    Return modes
    <x, y>     : x,y coordinates of the DM actuators. 
    Could come from the output of the function:
    x, y, mask = generateActuCoordinates(nx, ny, Nactu) 
    OR can be any x,y DM coordinates. 

    <L0>       : outer scale, in the same units as x and y.
    <filterPiston> : True/False, to filter the piston out or not


    example of use:
    L0 = 25

    nx, ny = 64, 64
    Nactu = 3228
    x, y, mask = generateActuCoordinates(nx, ny, Nactu)

    modes =  KLmodes(x, y, L0, True)
    '''
    Nactu = len(x)
    # Computation of the matrix of distances (squared !)
    print('Computing matrix of distances')
    mdist2 = (x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2
    # normalisation of distances with respect to pupil diameter
    D = x.ptp()
    mdist2 /= D**2
    print('Computing covariance matrix')
    if L0 == 0:
        kolmo = -0.5 * 6.88 * mdist2**(5. / 6)
    else:
        kolmo = -0.5 * rodconan(np.sqrt(mdist2), L0 / D)
    if filterPiston:
        print('Filtering piston out')
        P = computePistonFilteringMatrix(Nactu)
        kolmo = P.dot(kolmo.dot(P))
    print('Diagonalisation')
    # L'unite des valuers propres est en radians^2 a la longueur d'onde ou
    # est exprime r0, et sachant que tout est normalise pour (D/r0)=1.
    #
    singValues, U = np.linalg.eigh(kolmo)
    singValues = singValues / Nactu
    print('done')
    return U[:, ::-1], singValues[::-1]


def computePistonFilteringMatrix(n):
    '''
    Creates a sqaure matrix that removes (=filter out) piston mode.
    <n> : size of the matrix

    P = computePistonFilteringMatrix(64)
    '''
    P = np.full((n, n), -1. / n)
    for i in range(n):
        P[i, i] += 1.0
    return P


def compute_Caa(vv1, vv2, s1, s2, Cn2h, L0, alt1, alt2, altitude):
    """

    This function computes the covariance matrix between TWO wavefront-sensors,
    each pointing in some particular direction of a given Guide Star for a UNI turbulent
    layer at a given altitude. 

    ----------------
    --- Warning! ---
    ----------------

    The simulated guide star ---IS ALWAYS--- TT sensitive i.e if the GS is set at a finite altitude, it simulates a LGS with the perfect TT sensing capability.
    If needed, the TT filtering should be performed afterwards on the covariance matrix by applying an averaging slopes matrix for the given WFS (see computeTTFiltMatrix for TT filtering or Filt_Zer for filtering other Zernikes modes)
    
    <vv1>, <vv2> is the array (N,2) of the subaperture coordinates projected
         onto altitude layer. vv[:,0] is the x-coordinate, vv([:,1] is y.
         vv1 is for WFS number 1, vv2 for the other. It's up to the user to
         order the subapertures in the way he expects to find them in the
         final cov matrix.
    <s1>, <s2> are the sizes of the subapertures **in the pupil** (i.e.
         at ground level, i.e. the real physical size in the telescope
         pupil, not the one projected onto the layer) 
    <Cn2h> strengh of the layer: it must be equal to r0**(-5./3) with r0
        in metres. Initially this parameter was r0 but the Learn & Apply fit was
        non-linear. Now the Learn will be searching for <Cn2h> which makes
        it a linear fit and convergence is faster.
    <L0> the outer scale (in metres)
    <alt1>, <alt2>: altitude of the source (the Guide Star) of the WFS (metres). A value
        of alt==0 is interpreted as alt==infinity.       
    <altitude>: altitude of the layer (metres). Used to recompute the real ssp size at the given altitude.

    -------------------
     Example of use:
    -------------------

    # Generating Arbitrary ssp coordinates with a 14x14 SH on a 8m telescope:
    Nssp = 14
    x = np.linspace(-1,1,Nssp)
    x, y = np.meshgrid(x,x)
    r=np.sqrt(x*x+y*y)<1.1
    vv1 = np.array([x[r].flatten(), y[r].flatten()]).T
    vv1*=4
    # We shift the second WFS by 0.5m
    vv2 =vv1 + 0.5

    s1=0.5 # Physical size of the ssp in meters  of wfs #1
    s2=0.5 # Physical size of the ssp in meters  of wfs #2
    r0 = 0.5 # in meters
    Cn2h = r0**(5/3)
    L0 = 50 # in meters

    alt1 = 0 # GS altitude 0 meters == infinite GS == NGS
    alt2 = 0 # GS altitude 0 meters == infinite GS == NGS

    altitude = 3500 # Altitude of the turbulence layer in meters

    caa = compute_Caa(vv1, vv2, s1, s2, Cn2h, L0, alt1, alt2, altitude) # Computes the WFS covariance matrix

    # --------------------------------------------
    #       Optional modes filtering:
    # --------------------------------------------

    nbslopes = caa.shape[0] # x +y slopes = total nb of slopes
    FiltMatrix = computeTTFiltMatrix(nbslopes) # ex with TT filtering (use Filt_Zer for other modes filtering)
    
    # Case of Covmat of NGS/LGS:
    caaFilt = np.dot(filtTTMat, caa)    

    # Case of Covmat of LGS/LGS:
    caaFilt = np.dot(np.dot(filtTTMat, caa), filtTTMat.T)
    
    """

    # vv are the subaperture coordinates projected onto altitude layer
    vx1 = vv1[:,0]
    vy1 = vv1[:,1]
    vx2 = vv2[:,0]
    vy2 = vv2[:,1]

    # matrix of distances in x and y for all subapertures couples between 2 WFSs.
    vx = vx1[:, None] - vx2[None, :]
    vy = vy1[:, None] - vy2[None, :]
    s1_pup = s1
    s2_pup = s2
    # rescale the subaperture size the projected size of a subaperture at a given altitude layer
    if alt1 > 0:
       s1 = s1*(alt1-altitude)/alt1
    if alt2 > 0:
       s2 = s2*(alt2-altitude)/alt2
    nssp1 = vx1.shape[0] #number of sub-apertures of 1st WFS
    nssp2 = vx2.shape[0] #number of sub-apertures of 2nd WFS
    # test if the altitude layers is higher than the LGS altitude 
    if (s1 <= 0) or (s2 <= 0):
        return np.zeros((2*nssp1, 2*nssp2))

    # Faut calculer la covariance en altitude et la ramener dans le plan pupille
    ac = s1/2-s2/2
    ad = s1/2+s2/2
    bc = -s1/2-s2/2
    bd = -s1/2+s2/2

    # Caa x-x
    caa_xx = -DPHI(vx+ac,vy,L0) + DPHI(vx+ad,vy,L0) + DPHI(vx+bc,vy,L0) - DPHI(vx+bd,vy,L0)
    caa_xx /= 2.

    # Caa y-y
    caa_yy = -DPHI(vx,vy+ac,L0) + DPHI(vx,vy+ad,L0) + DPHI(vx,vy+bc,L0) - DPHI(vx,vy+bd,L0)
    caa_yy /= 2.

    if False:
        # Calcul du rico pour ameliorer l'exactitude du modele en xy
        # Bon on desactive quand meme... c'est idem a 1e-5 pres
        s0 = np.sqrt(s1**2+s2**2)   # size of the subaperture equivalent to a convolution by s1 and s2
        caa_xy = -DPHI(vx+s0/2,vy-s0/2,L0) + DPHI(vx+s0/2,vy+s0/2,L0) + DPHI(vx-s0/2,vy-s0/2,L0) - DPHI(vx-s0/2,vy+s0/2,L0)
        caa_xy /= 2.
        caa_xy /= 2.
        # et la, calcul d'une correction car le modele est loin d etre parfait ...
        r = np.max([s1,s2])/np.min([s1,s2])
        coeff = 1./(1-(1-1/r)**2)
        caa_xy /= coeff
    else:
        caa_xy = -DPHI(vx+s1/2,vy-s2/2,L0) + DPHI(vx+s1/2,vy+s2/2,L0) + DPHI(vx-s1/2,vy-s2/2,L0) - DPHI(vx-s1/2,vy+s2/2,L0)
        caa_xy /= 2.

    caa = np.zeros((2*nssp1, 2*nssp2))
    caa[:nssp1,         :nssp2] = caa_xx
    caa[:nssp1,         nssp2:nssp2*2] = caa_xy
    caa[nssp1:nssp1*2, :nssp2] = caa_xy
    caa[nssp1:nssp1*2, nssp2:nssp2*2] = caa_yy  
    #units
    k = 1.0/s1_pup/s2_pup
    lambda2 = (206265*0.5e-6/2/np.pi)**2
    caa *=k*lambda2*(np.abs(Cn2h))

    return caa





def macdo_x56(x,k=10):
    """   
    Computation of the Mc Donald function.

    f(x) = x**(5/6)*K_{5/6}(x)
    using a series for the esimation of K_{5/6}, taken from Rod Conan thesis :
    K_a(x)=1/2 \sum_{n=0}**\infty \frac{(-1)**n}{n!}
    \left(\Gamma(-n-a) (x/2)**{2n+a} + \Gamma(-n+a) (x/2)**{2n-a} \right) ,
    with a = 5/6.

    Setting x22 = (x/2)**2, setting uda = (1/2)**a, and multiplying by x**a,
    this becomes :
    x**a * Ka(x) = 0.5 $ -1**n / n! [ G(-n-a).uda x22**(n+a) + G(-n+a)/uda x22**n ]
    Then we use the following recurrence formulae on the following quantities :
    G(-(n+1)-a) = G(-n-a) / -a-n-1
    G(-(n+1)+a) = G(-n+a) /  a-n-1
    (n+1)! = n! * (n+1)
    x22**(n+1) = x22**n * x22
    and at each iteration on n, one will use the values already computed
    at step (n-1).
    The values of G(a) and G(-a) are hardcoded instead of being computed.

    The first term of the series has also been skipped, as it
    vanishes with another term in the expression of Dphi.
    
    """
    x = np.array(x) # Safe check
    a = 5./6.
    fn = 1.                             # initialisation factorielle 0!=1
    x2a = x**(2.*a)
    x22 = x*x/4.                        #  (x/2)**2
    x2n = 0.5                           # init (1/2) * x**0
    Ga  =  2.01126983599717856777       # Gamma(a) / (1/2)**a
    Gma = -3.74878707653729348337       # Gamma(-a) * (1/2.)**a
    s = np.zeros(x.shape)
    for n in range(k+1):
      dd = Gma * x2a
      if n:
        dd += Ga
      dd *= x2n
      dd /= fn
      # addition to s, with multiplication by (-1)**n
      if n%2:
          s -= dd
      else:
          s += dd
      # prepare recurrence iteration for next step
      if n<k:
        fn *= n+1     # factorial
        Gma /= -a-n-1 # gamma function
        Ga /= a-n-1   # idem
        x2n *= x22    # x**n
    return s



def asymp_macdo(x):
    """
    Computes a term involved in the computation of the phase struct
    function with a finite outer scale according to the Von-Karman
    model. The term involves the MacDonald function (modified bessel
    function of second kind) K_{5/6}(x), and the algorithm uses the
    asymptotic form for x ~ infinity.
    Warnings :
        - This function makes a floating point interrupt for x=0
    and should not be used in this case.
        - Works only for x>0.
    
    """
    x = np.array(x)
    # k2 is the value for
    # gamma_R(5./6)*2**(-1./6)
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081   #  sqrt(pi/2)
    a1 = 0.22222222222222222222   #  2/9
    a2 = -0.08641975308641974829  #  -7/89
    a3 = 0.08001828989483310284   # 175/2187
    x_1 = 1./x
    res = k2 - k3*np.exp(-x)*x**(1/3.)*(1.0 + x_1*(a1 + x_1*(a2 + x_1*a3)))
    return res




def rodconan(r,L0,k=10):
    """ DOCUMENT rodconan(r,L0,k=)
    The phase structure function is computed from the expression
    Dphi(r) = k1  * L0**(5./3) * (k2 - (2.pi.r/L0)**5/6 K_{5/6}(2.pi.r/L0))

    For small r, the expression is computed from a development of
    K_5/6 near 0. The value of k2 is not used, as this same value
    appears in the series and cancels with k2.
    For large r, the expression is taken from an asymptotic form.
    
    """
    # k1 is the value of :
    # 2*gamma_R(11./6)*2**(-5./6)*pi**(-8./3)*(24*gamma_R(6./5)/5.)**(5./6)
    k1 = 0.1716613621245709486
    dprf0 = (2*np.pi/L0)*r
    # k2 is the value for gamma_R(5./6)*2**(-1./6),
    # but is now unused
    # k2 = 1.0056349179985892838    
    res = np.zeros(r.shape)
    Xlim = 0.75*2*np.pi
    largeX = dprf0>Xlim

    res[largeX] = asymp_macdo(dprf0[largeX])
    smallX = np.logical_not(largeX)
    res[smallX] = -macdo_x56(dprf0[smallX], k=k)
    return (k1 * L0**(5./3)) * res 





def DPHI(x,y,L0):
    """ 
    dphi = DPHI(x,y,L0) * r0**(-5./3)

   Computes the phase structure function for a separation (x,y).
   The r0 is not taken into account : the final result of DPHI(x,y,L0)
   has to be scaled with r0**-5/3, with r0 expressed in meters, to get
   the right value.
    """

    r = np.sqrt(x**2+y**2)

    """  BEFORE ...... when rod conan did not exist
    fracDim = 5./3.    # Can vary fracDim for those who do not believe in Kolmogorov...
    r53 = r**(fracDim)
    return 6.88*r53
    """ 
    # With L0 ......
    return rodconan(r, L0)




def computeTTFiltMatrix(nbslopes):
    """
    Returns a matrix <p1> that filters the average TT on the vector of slopes <vec>
    with nbslopes = x+y slopes
    
    ---------------
    Example of use:
    ---------------

    nbslopes = 300
    p1 = computeTTFiltMatrix(nbslopes)
    vec_filt = np.dot(p1,vec)
        
    """
    p = filt_TTvec(nbslopes)
    Id = np.identity(nbslopes)
    p1 = Id-p
    return p1


def filt_TTvec(nbslopes):
    """
    Returns a matrix p that averages the slopes over x and y. 

    ---------------
    Example of use:
    ---------------

    nbslopes = 300
    p = filt_TTvec(nbslopes)
    vec_filtered = np.dot(p,vec)

    """
    p = np.zeros((nbslopes, nbslopes))
    p[0:nbslopes//2,0:nbslopes//2] = 1
    p[nbslopes//2:,nbslopes//2:] = 1
    p/=nbslopes/2

    return p


def Filt_Zer(modesfilt, miz):
    """
    MatFiltZer = Filt_Zer(modesfilt, miz);
    This function generates a matrix that filters all the zernikes modes up to <modesfilt>
    It requires a zernike interaction matrix <miz> of the given the WFS

    ---------------
    Example of use:
    ---------------

    modesfilt = 10
    MatFiltZer = Filt_Zer(modesfilt, miz)

    The filtered vector can be computed using:
    vec_filt = np.dot(MatFiltZer,miz)

    With <miz> computed using doit:
    from hraa.tools.doit import doit
    nssp = 14
    obs = 0
    nbzer = 100
    miz = doit(nssp, obs, nbzer)

    """
    mizFilt = miz[:,:modesfilt]
    mrz = np.linalg.pinv(mizFilt)
    MatFiltZer = np.identity(miz.shape[0]) - np.dot(mizFilt, mrz)
    return MatFiltZer