# -*- coding: utf-8 -*-t
"""
Created on Sun May 2 10:00:00 2021

@author: nonli
"""
import numpy as np
import tensorflow as tf
import math as m

from tensorflow.python.keras.regularizers import _none_to_default

# from tensorflow.keras import backend as K (to remove this line)
from thqml import phasespace as qx

# OMEGAR
# OMEGAI
# N
# n
# L
# Rq, Rp
# dtype
# chi
# DEFAULT_VALUES

# type of model (qs or nn)
model = "qs"
N = 20
n = 10
halfn = 5
L = 1
chi = 1
OMEGAI = None
OMEGAR = None
Rq = None
Rp = None
# define default parameters for AAH model
AAH_VALUES = {
    "k0": 0.0,
    "k1": 1.0,
    "tau": 1.0,
    "phi": 0.0,
}
cos_AAH = tf.constant(0.0, dtype=tf.double)
# J = None
J = tf.constant(0.0, dtype=tf.double)


# default np real type
np_default_real = np.float64

# default np complex type
np_default_complex = np.complex64

np_real = np_default_real
tf_real = tf.float32
dtype = tf_real

# default values for parameters
DEFAULT_VALUES = {
    "n": n,
    "L": L,
    "np_real": np_default_real,
    "np_complex": np_default_complex,
    "tf_real": tf_real,
    "dtype": tf_real,
    "chi": chi,
    "v_AAH": 0.0,
    "b_AAH": 1.0,
    "phi_AAH": 0.0,
    "model": "qs",
}


def init(input_dic, **kwargs):
    """init function that defines the global matrices"""
    global N, n, halfn, L, Rq, Rp, J, dtype, chi, model, OMEGAR, OMEGAI
    global v_AAH, b_AAH, phi_AAH, cos_AAH

    # import main parameters

    if "n" in input_dic:
        n = input_dic["n"]
    else:
        n = DEFAULT_VALUES["n"]

    if "L" in input_dic:
        L = input_dic["L"]
    else:
        L = DEFAULT_VALUES["L"]

    if "chi" in input_dic:
        chi = input_dic["chi"]
    else:
        chi = DEFAULT_VALUES["chi"]

    if "dtype" in input_dic:
        dtype = input_dic["dtype"]
    else:
        dtype = DEFAULT_VALUES["dtype"]

    # set main parameter
    N = 2 * n
    halfn = int(n / 2)
    Rq_np, Rp_np, J_np = qx.RQRP(N)
    Rq = tf.constant(Rq_np, dtype=dtype)
    Rp = tf.constant(Rp_np, dtype=dtype)
    J = tf.constant(J_np, dtype=dtype)

    # set AAH parameters
    if "v_AAH" in input_dic:
        v_AAH = input_dic["v_AAH"]
    else:
        v_AAH = DEFAULT_VALUES["v_AAH"]

    if "b_AAH" in input_dic:
        b_AAH = input_dic["b_AAH"]
    else:
        b_AAH = DEFAULT_VALUES["b_AAH"]

    if "phi_AAH" in input_dic:
        phi_AAH = input_dic["phi_AAH"]
        phi_AAH = phi_AAH * np.ones((n,), dtype=np_default_real)
    else:
        phi_AAH = np.zeros((n,), dtype=np_default_real)

    # set model dependent parameters
    if "model" in input_dic:
        model = input_dic["model"]
    else:
        model = DEFAULT_VALUES["model"]

    if model == "qs":
        OMEGAR, OMEGAI = evaluate_omega(dtype=dtype)
    elif model == "nnt":  # bose hubbard with nn non periodical
        OMEGAR, OMEGAI = evaluate_omega_nnt(dtype=dtype)
        print("Setting Bose-Hubbard model L=n (non periodical)")
        L = n
    elif model == "aah":
        OMEGAR, OMEGAI = evaluate_omega_nnt(dtype=dtype)
        print("Setting AAH model L=n (non periodical)")
        L = n
        # set the modulation tensor
        # coumpute the potential for the AAH model
        build_AAH_model(input_dic, **kwargs)
    #        phases = tf.constant(phi_AAH, dtype=dtype)
    #        cos_AAH=v_AAH*tf.math.cos(2.0*m.pi*b_AAH*(tf.range(n,dtype=dtype)+1)+phases)
    else:  # bose hubbard with nn
        OMEGAR, OMEGAI = evaluate_omega_nn(dtype=dtype)
        print("Setting Bose-Hubbard model L=n")
        L = n
    return None


def build_AAH_model(input_dic, **kwargs):
    """build parameters for AAH model"""
    global AAH_VALUES, cos_AAH

    # set AAH parameters
    if "k0_AAH" in input_dic:
        AAH_VALUES["k0"] = input_dic["k0_AAH"]

    if "k1_AAH" in input_dic:
        AAH_VALUES["k1"] = input_dic["k1_AAH"]

    if "phi_AAH" in input_dic:
        AAH_VALUES["phi"] = input_dic["phi_AAH"]

    if "tau_AAH" in input_dic:
        AAH_VALUES["tau"] = input_dic["tau_AAH"]

    # compute the AAH potential
    k0 = AAH_VALUES["k0"]
    k1 = AAH_VALUES["k1"]
    tau = AAH_VALUES["tau"]
    phi = AAH_VALUES["phi"]

    cos_AAH = k0 + k1 * tf.math.cos(2.0 * m.pi * (tf.range(n, dtype=dtype)) / tau + phi)

    return None


def compute_eigenvalues_AAH(**kwargs):
    """Return the eigenvalue of the AAH model"""
    AAH_np = OMEGAR.numpy() + np.diag(cos_AAH.numpy())

    return None


# def evaluate_omega(**kwargs):
#     """returns a constant tf tensor (n,n) with the kinetic matrix

#     Returns
#     -------
#     Omegar constant tensor
#     Omegai constant tensor

#     """
#     omega = np.zeros([n, n], dtype=np_default_complex)
#     for p in range(n):
#         for q in range(p, n):
#             somma = 0.0
#             for s in range(halfn):
#                 ksint = s
#                 ks = 2.0 * np.pi * ksint / L
#                 somma = somma + np.square(ks) * np.exp(
#                     1j * 2.0 * np.pi * (p - q) * s / n
#                 )
#             #                somma = somma+np.square(ks)*np.exp(1j*2.0*np.pi*(p-q)*ksint/n)
#             for s in range(halfn, n):
#                 ksint = s - n
#                 ks = 2.0 * np.pi * ksint / L
#                 somma = somma + np.square(ks) * np.exp(
#                     1j * 2.0 * np.pi * (p - q) * s / n
#                 )
#             #                somma = somma+np.square(ks)*np.exp(1j*2.0*np.pi*(p-q)*ksint/n)
#             omega[p, q] = somma / n
#             omega[q, p] = somma / n
#     return tf.constant(np.real(omega), **kwargs), tf.constant(np.imag(omega), **kwargs)


def evaluate_omega_nn(**kwargs):
    """returns constant tf tensor (n,n) k matrix nearest neiborhood periodical

    Returns
    -------
    Omegar constant tensor
    Omegai constant tensor

    """
    omega = np.zeros([n, n], dtype=np_default_complex)
    for p in range(n):
        pp1 = np.mod(p + 1, n)
        pm1 = np.mod(p - 1, n)
        omega[p, pp1] = -1.0
        omega[p, pm1] = -1.0
    return tf.constant(np.real(omega), **kwargs), tf.constant(np.imag(omega), **kwargs)


def evaluate_omega(**kwargs):
    """returns a constant tf tensor (n,n) with the kinetic matrix

    Returns
    -------
    Omegar constant tensor
    Omegai constant tensor

    """
    omega = np.zeros([n, n], dtype=np_default_complex)
    for p in range(n):
        for q in range(p, n):
            somma = 0.0
            for s in range(halfn):
                ksint = s
                ks = 2.0 * np.pi * ksint / L
                somma = somma + np.square(ks) * np.exp(
                    1j * 2.0 * np.pi * (p - q) * s / n
                )
            #                somma = somma+np.square(ks)*np.exp(1j*2.0*np.pi*(p-q)*ksint/n)
            for s in range(halfn, n):
                ksint = s - n
                ks = 2.0 * np.pi * ksint / L
                somma = somma + np.square(ks) * np.exp(
                    1j * 2.0 * np.pi * (p - q) * s / n
                )
            #                somma = somma+np.square(ks)*np.exp(1j*2.0*np.pi*(p-q)*ksint/n)
            omega[p, q] = somma / n
            omega[q, p] = somma / n
    return tf.constant(np.real(omega), **kwargs), tf.constant(np.imag(omega), **kwargs)


def evaluate_omega_nnt(**kwargs):
    """returns tf tensor (n,n) k matrix nearest neiborhood aperiodical

    Returns
    -------
    Omegar constant tensor
    Omegai constant tensor

    """

    # compute the complex omega matrix
    omega = np.zeros([n, n], dtype=np.complex64)
    for p in range(n):
        if p + 1 < n:
            pp1 = p + 1
            omega[p, pp1] = -1.0
        if p - 1 >= 0:
            pm1 = p - 1
            omega[p, pm1] = -1.0

    # return two constant tf.tensors
    # with real and imaginary part
    # **kwargs is used to pass some arguments from
    # input as the precision to be adopted
    tfomegar = tf.constant(np.real(omega), **kwargs)
    tfomegai = tf.constant(np.imag(omega), **kwargs)
    return tfomegar, tfomegai

@tf.function
def kinetic_energy(tensor, **kwargs):
    """Compute the kinetic energy by covariance, d and Omega matrix

    Parameters
    ----------
    tensor[0]: covariance matrix (N, N)
    tensor[1]: displacement vector <R> (1, N)

    Returns
    -------
    Ktotal  total kinetic energy (scalar)

    Remark
    ------
    N, n, OMEGAR and OMEGAI are global variablesglobal OMEGA

    """
    cov = tensor[0]
    displ = tensor[1]

    # compute the second derivatives
    djk = tf.squeeze(tf.tensordot(displ, displ, axes=0))
    #    chimn = tf.math.scalar_mul(-0.5, cov)+tf.math.scalar_mul(-1.0, djk) modified by claudio june
    chimn = tf.add(tf.math.scalar_mul(-0.5, cov), tf.math.scalar_mul(-1.0, djk))
    # build the tensor of mixed derivatives
    chi2R = tf.math.scalar_mul(-0.5, tf.eye(n, dtype=cov.dtype, **kwargs))
    #    chi2R = tf.zeros([n, n], dtype=cov.dtype, **kwargs)
    chi2I = tf.zeros([n, n], dtype=cov.dtype, **kwargs)
    for j in tf.range(n):
        qj = 2 * j
        pj = qj + 1
        tmpR = tf.gather_nd(chimn, [[qj, qj]]) + tf.gather_nd(chimn, [[pj, pj]])
        tmpR = -0.5 * tmpR
        chi2R = tf.tensor_scatter_nd_add(chi2R, [[j, j]], tmpR)
        for k in tf.range(j + 1, n):
            qk = 2 * k
            pk = qk + 1
            tmpR = tf.gather_nd(chimn, [[qj, qk]]) + tf.gather_nd(chimn, [[pj, pk]])
            tmpR = -0.5 * tmpR
            tmpI = tf.gather_nd(chimn, [[qj, pk]]) - tf.gather_nd(chimn, [[pj, qk]])
            tmpI = -0.5 * tmpI
            # build symmetric part
            chi2R = tf.tensor_scatter_nd_add(chi2R, [[j, k]], tmpR)
            chi2R = tf.tensor_scatter_nd_add(chi2R, [[k, j]], tmpR)
            # build antisymmetric
            chi2I = tf.tensor_scatter_nd_add(chi2I, [[j, k]], tmpI)
            chi2I = tf.tensor_scatter_nd_add(chi2I, [[k, j]], -tmpI)

    # compute the total kinetic energy
    ktotal = tf.math.reduce_sum(
        tf.add(tf.multiply(OMEGAR, chi2R), tf.multiply(OMEGAI, chi2I))
    )
    return ktotal


@tf.function
def gaussian_boson_numbers(tensor, **kwargs):
    """Function that returns n,n2, and Dn2 given g,d,hessian
    (used in variational ansatz)

    In the call

    Parameters
    ----------
    tensor[0]: covariance matrix (N,N)
    tensor[1]: avg displacement vector (1,N)
    tensor[2]: hessian matrix (N,N)

    g=tensor[0]
    d=tensor[1]
    hessian=tensor[2]

    Returns
    -------
    param: nboson: number of bosons per mode (1,n)
    param: nboson2: squared number of bosons per mode (1,n)

    Remark
    ------
    N, Rq, Rp are global here


    """
    g = tensor[0]
    d = tensor[1]
    hessian = tensor[2]

    # evaluate the laplacian by gathering the diagonal elements of gaussian
    cr_2x = tf.reshape(tf.linalg.diag_part(hessian), [1, N])

    # sum the derivatives to have the squared laplacian
    dqq = tf.matmul(cr_2x, Rq)  # [1,N]
    dpp = tf.matmul(cr_2x, Rp)  # [1,N]
    laplacian = dqq + dpp  # [1,N]

    # evaluate the biharmonic by diag g and d
    d = tf.squeeze(d)  # [N,] dj
    gd = tf.linalg.diag_part(g)  # [N,] gjj
    d2 = tf.square(d)  # [N,] dj^2
    d4 = tf.tensordot(d2, d2, axes=0)  # [N,N] dj^2 dk^2
    dd = tf.tensordot(d, d, axes=0)  # [N,N] dj dk

    djjkk = (
        0.25 * tf.tensordot(gd, gd, axes=0)
        + 0.5 * tf.square(g)
        + 0.5 * tf.tensordot(gd, d2, axes=0)
        + 0.5 * tf.tensordot(d2, gd, axes=0)
        + 2.0 * tf.multiply(g, dd)
        + d4
    )

    biharmonic = tf.zeros_like(laplacian)  # [1,N]
    for j in tf.range(n):
        qj = 2 * j
        pj = 2 * j + 1
        dqqpp = tf.gather_nd(djjkk, [[qj, pj]])
        dqqqq = tf.gather_nd(djjkk, [[qj, qj]])
        dpppp = tf.gather_nd(djjkk, [[pj, pj]])
        biharmonic = tf.tensor_scatter_nd_add(
            biharmonic, [[0, j]], dqqqq + dpppp + 2 * dqqpp
        )
    # evaluate the number of bosons
    nboson = -0.5 * laplacian - 0.5

    # evaluate the n2
    nboson2 = 0.25 * biharmonic + 0.5 * laplacian

    return nboson, nboson2


@tf.function
def potential_energy_AAH(tensor, **kwargs):
    """Return the potential energy for the AAH model

    TODO move the parameters in the AAH definition

    Parameters
    ----------
    tensor[0] number of bosons (1,n)
    tensor[1] number of bosons squared (1,n)

    Returns
    -------
    Vtotal: total potential energy, global variable

    Remark, chi is a global variable here
    """
    nb = tensor[0]
    nb2 = tensor[1]

    # potential energy (chi/dx) sum (nj^2 - nj) # check sign here
    pe = tf.math.scalar_mul(0.5 * chi * n / L, tf.math.reduce_sum(nb2 - nb))
    # add the cosin part
    pe = pe + tf.math.reduce_sum(tf.multiply(nb, cos_AAH))
    return pe


@tf.function
def potential_energy(tensor, **kwargs):
    """
    Return the potential energy


    Parameters
    ----------
    tensor[0] number of bosons (1,n)
    tensor[1] number of bosons squared (1,n)

    Returns
    -------
    Vtotal: total potential energy, global variable

    Remark, chi is a global variable here
    """
    nb = tensor[0]
    nb2 = tensor[1]

    # potential energy (chi/dx) sum (nj^2 - nj)
    pe = tf.math.scalar_mul(0.5 * chi * n / L, tf.math.reduce_sum(nb2 - nb))
    return pe


@tf.function
def total_energy(tensor, **kwargs):
    """Return the potential energy

    Parameters
    ----------
    tensor[0] kinetic energy (scalar)
    tensor[1] potential energy (scalar)

    Returns
    -------
    Vtotal: total potential energy

    """
    kt = tensor[0]
    vt = tensor[1]
    return kt + vt


@tf.function
def nonlinear_eigenvalue(tensor, **kwargs):
    """Return the nonlinear eigenvalue

    Parameters
    ----------
    tensor[0] hamiltonian H (scalar)
    tensor[1] total particle N (scalar)

    Returns
    -------
    EH: eigenvalue as H/N

    Remark, chi is a global variable here
    """
    ht = tensor[0]
    nt = tensor[1]
    return ht / nt


class LyapunovLayer(tf.keras.layers.Layer):
    """Return the Lyapunov functional as F=H-beta N with beta trainable variable

    In the creation F=Lyapunov(beta) with beta initial value for the beta

    In the call, F(tensors)
    In input tensors, with H=tensors[0] and N=tensors[1]

    """

    def __init__(self, beta_np=1.0, **kwargs):
        super(LyapunovLayer, self).__init__(**kwargs)
        self.beta = tf.Variable(beta_np, dtype=self.dtype)

    def get_beta(self):
        return self.beta

    def call(self, inputs):
        Htmp = inputs[0]
        Ntmp = inputs[1]
        return Htmp - self.beta * Ntmp


def entanglement_entropy(cov_np):
    """np function for computing entanglement entropy from covariance"""
    Jnp = J.numpy()
    # attenzione forse qui errore
    #    HG = np.matmul(Jnp.transpose(),(0.5*cov_np+0.5*1j*Jnp))
    #    HG = np.matmul(Jnp.transpose(),(0.5*cov_np+1j*Jnp)) # check this
    HG = np.matmul(Jnp.transpose(), (0.5 * cov_np))  # check this
    e1, _ = np.linalg.eig(HG)
    e1 = np.abs(e1)
    e1red = np.sort(np.sort(e1)[0::2])
    eeout = 0.0
    for ij in tf.range(n):
        etmp = e1red[ij]
        if etmp > 0.5:
            tmpp = (etmp + 0.5) * np.log2(etmp + 0.5)
            tmpm = (etmp - 0.5) * np.log2(etmp - 0.5)
            tmp = tmpp - tmpm
            eeout = eeout + tmp
        elif etmp < 0.49:  # 0.5 with tolerance
            print("Warning, entropy, unexpected eig " + repr(etmp))
    return eeout


def logarithmic_negativity(cov_np, mask=np.zeros(n)):
    """Return the entropy of entanglement as determined from the eigs of cov

    Reference: vidal, PRA 65, 032314 (2002)

    cov: is the transposed cov matrix according to the mask

    the logarithmic negativity is computed by the symplectic eigenvalues cj as
    EN=sum_j=0^(n-1) F(cj)
    with F(cj)=0 if c>=0.5 and F(cj)=-log2(2.0*cj) for c<0.5

    Input:
    ------
    cov, covariance matrix tensor [N,N]
    mask, np array mask transpose [n,] (one for the modes to transpose)


    Output:
    -------
    EN , logarithmic negativity
    negativity, negativity as EN=log2(2 N+1), or N=0.5*(2^EN-1)
    symplectic_eigs, symplectic eigenvalues
    """

    # build a diagonal with -1 corresponding to the momenta to be transposed
    diag1 = np.ones(N, dtype=np_real)
    for j in range(n):
        if mask[j] == 1:
            diag1[2 * j] = 1
            diag1[2 * j + 1] = -1
            # build the transformation matrix
    Ftrans = np.diag(diag1)
    # transform the covariance matrix
    cov_np_Ta = np.matmul(Ftrans, np.matmul(cov_np, Ftrans))

    Jnp = J.numpy()
    HG = np.matmul(Jnp.transpose(), (0.5 * cov_np_Ta))
    e1, _ = np.linalg.eig(HG)
    e2, v2 = np.linalg.eig(-np.matmul(HG, HG))

    ae1 = np.abs(e1)
    e1red = np.sort(np.sort(ae1)[0::2])

    ENeg = 0.0
    for ij in tf.range(n):
        etmp = e1red[ij]
        if etmp < 0.5:
            ENeg = ENeg - np.log2(2.0 * etmp)

    negativity = 0.5 * (np.power(2, ENeg) - 1.0)

    return ENeg, negativity, e1red, e2, v2

def logarithmic_negativity_fast(cov_np, mask=np.zeros(n)):
    """Return the entropy of entanglement as determined from the eigs of cov, fast verison

    Reference: vidal, PRA 65, 032314 (2002)

    cov: is the transposed cov matrix according to the mask

    the logarithmic negativity is computed by the symplectic eigenvalues cj as
    EN=sum_j=0^(n-1) F(cj)
    with F(cj)=0 if c>=0.5 and F(cj)=-log2(2.0*cj) for c<0.5

    Input:
    ------
    cov, covariance matrix tensor [N,N]
    mask, np array mask transpose [n,] (one for the modes to transpose)



    Output:
    -------
    EN , logarithmic negativity
    negativity, negativity as EN=log2(2 N+1), or N=0.5*(2^EN-1)
    symplectic_eigs, symplectic eigenvalues
    """

    # build a diagonal with -1 corresponding to the momenta to be transposed
    diag1 = np.ones(N, dtype=np_real)
    for j in range(n):
        if mask[j] == 1:
            diag1[2 * j] = 1
            diag1[2 * j + 1] = -1
            # build the transformation matrix
    Ftrans = np.diag(diag1)
    # transform the covariance matrix
    cov_np_Ta = np.matmul(Ftrans, np.matmul(cov_np, Ftrans))

    # compute symplectic eigenvalues
    Jnp = J.numpy()
    HG = np.matmul(Jnp.transpose(), (0.5 * cov_np_Ta))
    e1, _ = np.linalg.eig(HG)

    ae1 = np.abs(e1)
    e1red = np.sort(np.sort(ae1)[0::2])

    ENeg = 0.0
    for ij in tf.range(n):
        etmp = e1red[ij]
        if etmp < 0.5:
            ENeg = ENeg - np.log2(2.0 * etmp)

    return ENeg


def soliton_model(**kwargs):
    """create a soliton model"""

    # create layer with random init
    vacuum = qx.VacuumLayer(N, dtype=tf_real, name="Vacuum")
    dtarget = 0.1 * np.ones([N, 1], dtype=np_real)
    D = qx.TrainableDisplacementLayer(dtarget, dtype=tf_real, name="Displacement")

    CovL = qx.CovarianceLayer(N, name="covariance", dtype=tf_real)
    KinL = tf.keras.layers.Lambda(kinetic_energy, name="K", dtype=tf_real)
    BL = tf.keras.layers.Lambda(gaussian_boson_numbers, name="Bosons", dtype=tf_real)
    NL = tf.keras.layers.Lambda(tf.reduce_sum, dtype=tf_real, name="N")
    VL = tf.keras.layers.Lambda(potential_energy, dtype=tf_real, name="V")
    HL = tf.keras.layers.Lambda(total_energy, dtype=tf_real, name="H")

    # Create the model by pull back
    xin = tf.keras.layers.Input(N, dtype=tf_real, name="Input")
    x1 = xin
    a1 = None
    x1, a1 = qx.RandomLayer(N, dtype=tf_real, name="R")(x1, a1)
    x1, a1 = D(x1, a1)
    for ij in range(n):
        r_np = 0.1 * np.random.rand(1)
        n_squeeze = ij
        theta_np = 2.0 * np.pi * np.random.rand(1)
        x1, a1 = qx.SingleModeSqueezerLayer(
            N,
            r_np=r_np,
            theta_np=theta_np,
            n_squeezed=n_squeeze,
            trainable=True,
            dtype=tf_real,
            name="S" + repr(ij),
        )(x1, a1)
    chir, chii = vacuum(x1, a1)
    PSImodel = tf.keras.Model(inputs=xin, outputs=[chir, chii])

    # model with observables
    cov, d, hessian = CovL(chir, chii, PSImodel)
    Ktotal = KinL([cov, d])
    [nboson, nboson2] = BL([cov, d, hessian])
    Ntotal = NL(nboson)
    Vtotal = VL([nboson, nboson2])
    Htotal = HL([Ktotal, Vtotal])

    # model for outputs
    outputs = [nboson, nboson2, Htotal, Vtotal, Ktotal, Ntotal, cov, d]
    #    outputs = cov
    model1 = tf.keras.Model(inputs=xin, outputs=outputs)

    return model1, Htotal, Ntotal, nboson


def twin_solitons_entanglement(
    nA, nB, Nexp=10, nepochs=60000, accuracy=1.0e-3, verbose=1, **kwargs
):
    """Compute entanglement of twin solitons by training a model

    Input:
    ------
    nA, position of peak A
    nB, position of peak B
    Nexp, total number of bosons


    """

    if Nexp == 0:
        raise Exception("N cannot be 0")

    # define the model
    twin_model, Htotal, Ntotal, nboson = soliton_model()

    # define the cost functions
    expH = tf.exp(Htotal / n)
    nAtf = tf.gather_nd(nboson, [[0, nA]])[0]
    nBtf = tf.gather_nd(nboson, [[0, nB]])[0]
    expN = tf.exp(-nAtf)
    expDN = tf.square(nAtf - nBtf)
    Nexp_tf = tf.constant(Nexp, dtype=tf_real)

    # compute the entropy
    twin_model.add_loss(tf.reduce_mean(tf.square(Ntotal - Nexp_tf)))
    twin_model.add_loss(expH)
    twin_model.add_loss(expN)
    twin_model.add_loss(expDN)
    twin_model.compile(optimizer="adam")

    if verbose > 0:
        print("Training model for epochs " + repr(nepochs) + " ...")

    # dummy training points
    Nbatch = 1
    xtrain = np.random.rand(Nbatch, N) - 0.5
    ytrain = np.zeros([Nbatch, 1])
    for i1 in range(Nbatch):
        ytrain[i1] = 0

    # fit model
    history = twin_model.fit(xtrain, ytrain, epochs=nepochs, verbose=0)
    if verbose > 0:
        print("Initial loss " + repr(history.history["loss"][0]))
        print("  Final loss " + repr(history.history["loss"][-1]))

    # get output of trained model with dummy input
    results = twin_model(np.random.rand(1, N) - 0.5)
    Nout_final = results[5].numpy()
    if verbose > 0:
        print("Number of total bosons after training " + repr(Nout_final) + " ...")

    # check accuracy
    if 1 - abs(Nout_final / Nexp) > accuracy:
        print("Warning, low accuracy  ******************* !")
    # compute covariance matrix
    cov_np = results[6].numpy()

    # mask for negativity
    nmask = np.zeros(n)
    nmask[nA] = 1

    # # compute negativity
    ENeg = logarithmic_negativity_fast(cov_np, nmask)

    # write a line before exiting
    if verbose > 0:
        print(" ################################### ")

    return ENeg, twin_model
