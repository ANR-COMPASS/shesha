## @package   shesha.util.iterkolmo
## @brief     Stencil and matrices computation for the creation of a turbulent screen
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.5.0
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2023 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.

import numpy as np


def create_stencil(n):
    """ TODO: docstring
    """
    Zx = np.array(np.tile(np.arange(n), (n, 1)), dtype=np.float64) + 1
    Zy = Zx.T

    Xx = np.array(np.zeros(n) + n + 1, dtype=np.float64)
    Xy = np.array(np.arange(n) + 1, dtype=np.float64)

    ns = int(np.log2(n + 1) + 1)
    stencil = np.zeros((n, n))

    stencil[:, 0] = 1
    for i in range(2, ns):
        stencil[::2**(i - 1), 2**(i - 2)] = 1
        stencil.itemset(2**(i - 1) - 1, 1)
        # stencil[0,2**(i-1)-1]=1

    # i=ns
    stencil.itemset(2**(ns - 1) - 1, 1)
    for i in range(0, n, 2**(ns - 1)):
        stencil.itemset(2**(ns - 2) + i * n, 1)
    # i=ns+1
    stencil.itemset(2**(ns) - 1, 1)
    for i in range(0, n, 2**(ns)):
        stencil.itemset(2**(ns - 1) + i * n, 1)

    stencil = np.roll(stencil, n // 2, axis=0)
    stencil = np.fliplr(stencil)

    istencil = np.where(stencil.flatten() != 0)[0]

    return Zx, Zy, Xx, Xy, istencil


def stencil_size(n):
    """ TODO: docstring
    """
    stencil = np.zeros((n, n))
    ns = int(np.log2(n + 1) + 1)

    stencil[:, 0] = 1
    for i in range(2, ns):
        stencil[::2**(i - 1), 2**(i - 2)] = 1
        stencil.itemset(2**(i - 1) - 1, 1)

    # i=ns
    stencil.itemset(2**(ns - 1) - 1, 1)
    for i in range(0, n, 2**(ns - 1)):
        stencil.itemset(2**(ns - 2) + i * n, 1)
    # i=ns+1
    stencil.itemset(2**(ns) - 1, 1)
    for i in range(0, n, 2**(ns)):
        stencil.itemset(2**(ns - 1) + i * n, 1)

    return np.sum(stencil)


def stencil_size_array(size):
    """ Compute_size2(np.ndarray[ndim=1, dtype=np.int64_t] size)

    Compute the size of a stencil, given the screen size

    Args:

        size: (np.ndarray[ndim=1,dtype=np.int64_t]) :screen size
    """
    stsize = np.zeros(len(size), dtype=np.int64)

    for i in range(len(size)):
        stsize[i] = stencil_size(size[i])
    return stsize


def Cxz(n, Zx, Zy, Xx, Xy, istencil, L0):
    """ Cxz computes the covariance matrix between the new phase vector x (new
    column for the phase screen), and the already known phase values z.

    The known values z are the values of the phase screen that are pointed by
    the stencil indexes (istencil)
    """

    size = Xx.shape[0]
    size2 = istencil.shape[0]

    Xx_r = np.resize(Xx, (size2, size))
    Xy_r = np.resize(Xy, (size2, size))
    Zx_s = Zx.flatten()[istencil]
    Zy_s = Zy.flatten()[istencil]
    Zx_r = np.resize(Zx_s, (size, size2))
    Zy_r = np.resize(Zy_s, (size, size2))

    tmp = phase_struct((Zx[0, size - 1] - Xx)**2 + (Zy[0, size - 1] - Xy)**2, L0)
    tmp2 = phase_struct((Zx[0, size - 1] - Zx_s)**2 + (Zy[0, size - 1] - Zy_s)**2, L0)

    xz = -phase_struct((Xx_r.T - Zx_r)**2 + (Xy_r.T - Zy_r)**2, L0)

    xz += np.resize(tmp, (size2, size)).T + np.resize(tmp2, (size, size2))

    xz *= 0.5

    return xz


def Cxx(n, Zxn, Zyn, Xx, Xy, L0):
    """ Cxx computes the covariance matrix of the new phase vector x (new
   column for the phase screen).
    """
    size = Xx.shape[0]
    Xx_r = np.resize(Xx, (size, size))
    Xy_r = np.resize(Xy, (size, size))

    tmp = np.resize(phase_struct((Zxn - Xx)**2 + (Zyn - Xy)**2, L0), (size, size))

    xx = -phase_struct((Xx_r - Xx_r.T)**2 + (Xy_r - Xy_r.T)**2, L0)  # + \
    #    tmp+tmp.T

    xx += tmp + tmp.T

    xx *= 0.5

    return xx


def Czz(n, Zx, Zy, ist, L0):
    """ Czz computes the covariance matrix of the already known phase values z.

   The known values z are the values of the phase screen that are pointed by
   the stencil indexes (istencil)
   """

    size = ist.shape[0]
    Zx_s = Zx.flatten()[ist]
    Zx_r = np.resize(Zx_s, (size, size))
    Zy_s = Zy.flatten()[ist]
    Zy_r = np.resize(Zy_s, (size, size))

    tmp = np.resize(
            phase_struct((Zx[0, n - 1] - Zx_s)**2 + (Zy[0, n - 1] - Zy_s)**2, L0),
            (size, size))

    zz = -phase_struct((Zx_r - Zx_r.T) ** 2 + (Zy_r - Zy_r.T) ** 2, L0) + \
        tmp + tmp.T

    zz *= 0.5

    return zz


def AB(n, L0, deltax, deltay, rank=0):
    """ DOCUMENT AB, n, A, B, istencil
    This function initializes some matrices A, B and a list of stencil indexes
    istencil for iterative extrusion of a phase screen.

    The method used is described by Fried & Clark in JOSA A, vol 25, no 2, p463, Feb 2008.
    The iteration is :
    x = A(z-zRef) + B.noise + zRef
    with z a vector containing "old" phase values from the initial screen, that are listed
    thanks to the indexes in istencil.

    SEE ALSO: extrude createStencil Cxx Cxz Czz
    """

    #   if (rank == 0):
    #        print("create stencil and Z,X matrices")
    Zx, Zy, Xx, Xy, istencil = create_stencil(n)
    #    if (rank == 0):
    #        print("create zz")
    zz = Czz(n, Zx, Zy, istencil, L0)
    #    if (rank == 0):
    #        print("create xz")
    xz = Cxz(n, Zx, Zy, Xx, Xy, istencil, L0)
    #    if (rank == 0):
    #        print("create xx")
    xx = Cxx(n, Zx[0, n - 1], Zy[0, n - 1], Xx, Xy, L0)

    U, s, V = np.linalg.svd(zz)
    s1 = s
    s1[s.size - 1] = 1
    s1 = 1. / s1
    s1[s.size - 1] = 0
    zz1 = np.dot(np.dot(U, np.diag(s1)), V)
    #    if (rank == 0):
    #        print("compute zz pseudo_inverse")
    # zz1=np.linalg.pinv(zz)

    #    if (rank == 0):
    #        print("compute A")
    A = np.dot(xz, zz1)

    #    if (rank == 0):
    #        print("compute bbt")
    bbt = xx - np.dot(A, xz.T)
    #    if (rank == 0):
    #        print("svd of bbt")
    U1, l, V1 = np.linalg.svd(bbt)
    #    if (rank == 0):
    #        print("compute B")
    B = np.dot(U1, np.sqrt(np.diag(l)))

    test = np.zeros((n * n), np.float32)
    test[istencil] = np.arange(A.shape[1]) + 1
    test = np.reshape(test, (n, n), "C")
    isty = np.argsort(test.T.flatten("C")).astype(np.uint32)[n * n - A.shape[1]:]

    if (deltay < 0):
        isty = (n * n - 1) - isty
    if (deltax < 0):
        istencil = (n * n - 1) - istencil

    return np.asfortranarray(A.astype(np.float32)), np.asfortranarray(
            B.astype(np.float32)), istencil.astype(np.uint32), isty.astype(np.uint32)


def extrude(p, r0, A, B, istencil):
    """ DOCUMENT p1 = extrude(p,r0,A,B,istencil)

    Extrudes a phase screen p1 from initial phase screen p.
    p1 prolongates p by 1 column on the right end.
    r0 is expressed in pixels

    The method used is described by Fried & Clark in JOSA A, vol 25, no 2, p463, Feb 2008.
    The iteration is :
    x = A(z-zRef) + B.noise + zRef
    with z a vector containing "old" phase values from the initial screen, that are listed
    thanks to the indexes in istencil.

    Examples
    n = 32;
    AB, n, A, B, istencil;
    p = array(0.0,n,n);
    p1 = extrude(p,r0,A,B,istencil);
    pli, p1

    SEE ALSO: AB() createStencil() Cxx() Cxz() Czz()
    """

    amplitude = r0**(-5. / 6)
    n = p.shape[0]
    z = p.flatten()[istencil]
    zref = p[0, n - 1]
    z -= zref
    newColumn = np.dot(A, z) + np.dot(B, np.random.normal(0, 1, n) * amplitude) + zref
    p1 = np.zeros((n, n), dtype=np.float32)
    p1[:, 0:n - 1] = p[:, 1:]
    p1[:, n - 1] = newColumn

    return p1


def phase_struct(r, L0=None):
    """ TODO: docstring
    """
    if L0 is None:
        return 6.88 * r**(5. / 6.)
    else:
        return rodconan(np.sqrt(r), L0)


def rodconan(r, L0):
    """ The phase structure function is computed from the expression
     Dphi(r) = k1  * L0^(5./3) * (k2 - (2.pi.r/L0)^5/6 K_{5/6}(2.pi.r/L0))

     For small r, the expression is computed from a development of
     K_5/6 near 0. The value of k2 is not used, as this same value
     appears in the series and cancels with k2.
     For large r, the expression is taken from an asymptotic form.
    """

    # k1 is the value of :
    # 2*gamma_R(11./6)*2^(-5./6)*pi^(-8./3)*(24*gamma_R(6./5)/5.)^(5./6);
    k1 = 0.1716613621245709486
    dprf0 = (2 * np.pi / L0) * r

    Xlim = 0.75 * 2 * np.pi
    ilarge = np.where(dprf0 > Xlim)
    ismall = np.where(dprf0 <= Xlim)

    res = r * 0.
    """
    # TODO  those lines have been changed (cf trunk/yoga_ao/yorick/yoga_turbu.i l 258->264)
    if((ilarge[0].size > 0)and(ismall[0].size == 0)):
        res[ilarge] = asymp_macdo(dprf0[ilarge])
        # print("simulation atmos with asymptotic MacDonald function")
    elif((ismall[0].size > 0)and(ilarge[0].size == 0)):
        res[ismall] = -macdo_x56(dprf0[ismall])
        # print("simulation atmos with x56 MacDonald function")
    elif((ismall[0].size > 0)and(ilarge[0].size > 0)):
        res[ismall] = -macdo_x56(dprf0[ismall])
        res[ilarge] = asymp_macdo(dprf0[ilarge])
    """
    if (ilarge[0].size > 0):
        res[ilarge] = asymp_macdo(dprf0[ilarge])
    if (ismall[0].size > 0):
        res[ismall] = -macdo_x56(dprf0[ismall])

    return k1 * L0**(5. / 3.) * res


def asymp_macdo(x):
    """ Computes a term involved in the computation of the phase struct
     function with a finite outer scale according to the Von-Karman
     model. The term involves the MacDonald function (modified bessel
     function of second kind) K_{5/6}(x), and the algorithm uses the
     asymptotic form for x ~ infinity.

     Warnings :

         - This function makes a floating point interrupt for x=0
           and should not be used in this case.

         - Works only for x>0.
    """
    # k2 is the value for
    # gamma_R(5./6)*2^(-1./6)
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081  # sqrt(pi/2)
    a1 = 0.22222222222222222222  # 2/9
    a2 = -0.08641975308641974829  # -7/89
    a3 = 0.08001828989483310284  # 175/2187
    x_1 = 1. / x
    res = k2 - k3 * np.exp(-x) * x**(1 / 3.) * (1.0 + x_1 * (a1 + x_1 * (a2 + x_1 * a3)))
    return res


def macdo_x56(x, k=10):
    """ Computation of the function
    f(x) = x^(5/6)*K_{5/6}(x)
    using a series for the esimation of K_{5/6}, taken from Rod Conan thesis :
    K_a(x)=1/2 \\sum_{n=0}^\\infty \\frac{(-1)^n}{n!}
    \\left(\\Gamma(-n-a) (x/2)^{2n+a} + \\Gamma(-n+a) (x/2)^{2n-a} \\right) ,
    with a = 5/6.

    Setting x22 = (x/2)^2, setting uda = (1/2)^a, and multiplying by x^a,
    this becomes :
    x^a * Ka(x) = 0.5 $ -1^n / n! [ G(-n-a).uda x22^(n+a) + G(-n+a)/uda x22^n ]
    Then we use the following recurrence formulae on the following quantities :
    G(-(n+1)-a) = G(-n-a) / -a-n-1
    G(-(n+1)+a) = G(-n+a) /  a-n-1
    (n+1)! = n! * (n+1)
    x22^(n+1) = x22^n * x22
    and at each iteration on n, one will use the values already computed at step (n-1).
    The values of G(a) and G(-a) are hardcoded instead of being computed.

    The first term of the series has also been skipped, as it
    vanishes with another term in the expression of Dphi.
    """

    a = 5. / 6.
    fn = 1.  # initialisation factorielle 0!=1
    x2a = x**(2. * a)
    x22 = x * x / 4.  # (x/2)^2
    x2n = 0.5  # init (1/2) * x^0
    Ga = 2.01126983599717856777  # Gamma(a) / (1/2)^a
    Gma = -3.74878707653729348337  # Gamma(-a) * (1/2.)^a
    s = np.zeros(x.shape)  # array(0.0, dimsof(x));
    for n in range(k + 1):  # (n=0; n<=k; n++) {
        dd = Gma * x2a
        if n:
            dd += Ga
        dd *= x2n
        dd /= fn
        # addition to s, with multiplication by (-1)^n
        if (n % 2):
            s -= dd
        else:
            s += dd
        # prepare recurrence iteration for next step
        if (n < k):
            fn *= n + 1  # factorial
            Gma /= -a - n - 1  # gamma function
            Ga /= a - n - 1  # idem
            x2n *= x22  # x^n

    return s


def create_screen_assist(screen_size, L0, r0):
    """
    screen_size : screen size (in pixels)
    L0 : L0 in pixel
    r0 : total r0 @ 0.5 microns
    """
    A, B, istx, isty = AB(screen_size, L0)
    phase = np.zeros((screen_size, screen_size))

    print(stencil_size(screen_size))

    # pl.ion()
    # pl.imshow(phase, animated=True)
    # pl.show()

    for i in range(2 * screen_size):
        phase = extrude(phase, r0, A, B, istx)
        # pl.clf()
        # pl.imshow(phase, cmap='Blues')
        # pl.draw()

    return phase


def create_screen(r0, pupixsize, screen_size, L0, A, B, ist):
    """ DOCUMENT create_screen
        screen = create_screen(r0,pupixsize,screen_size,&A,&B,&ist)

        creates a phase screen and fill it with turbulence
        r0          : total r0 @ 0.5m
        pupixsize   : pupil pixel size (in meters)
        screen_size : screen size (in pixels)
        A           : A array for future extrude
        B           : B array for future extrude
        ist         : istencil array for future extrude
     """

    # AB, screen_size, A, B, ist,L0   # initialisation for A and B matrices for phase extrusion
    screen = np.zeros((screen_size, screen_size),
                      dtype=np.float32)  # init of first phase screen
    for i in range(2 * screen_size):
        screen = extrude(screen, r0 / pupixsize, A, B, ist)

    return screen
