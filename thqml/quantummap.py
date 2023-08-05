# -*- coding: utf-8 -*-t
"""
Created on 24 January 2022

Module for implementing qbits gates and quantum feature maps
with tensorflow

@author: nonlinearxwaves@gmail
"""

import tensorflow as tf
import numpy as np

# data type
tfcomplex = tf.complex64
npcomplex = np.complex64

# single 0 qubit
qubit0 = tf.constant([1, 0], dtype=tfcomplex)

# single 1 qubit
qubit1 = tf.constant([0, 1], dtype=tfcomplex)

# double 00 qubit
q00 = tf.tensordot(qubit0, qubit0, axes=0)

# double 11 qubit
q11 = tf.tensordot(qubit1, qubit1, axes=0)

# double 01 qubit
q01 = tf.tensordot(qubit0, qubit1, axes=0)

# double 10 qubit
q10 = tf.tensordot(qubit1, qubit0, axes=0)


# Hadamard gate for the single qbit
Hsingle = tf.constant(
    [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
    dtype=tfcomplex,
)

# Single Q-bit identity
Isingle = tf.constant([[1, 0], [0, 1]], dtype=tfcomplex)

# Single Q-bit Z gate
Zsingle = tf.constant([[1, 0], [0, -1]], dtype=tfcomplex)

# X gate for the single qbit
Xsingle = tf.constant([[0, 1], [1, 0]], dtype=tfcomplex)

# Y gate for the single qbit
Ysingle = tf.constant(
    np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex64), dtype=tfcomplex
)

# Two qbuits gates

# Identity gate for the two qubits
II = tf.tensordot(Isingle, Isingle, axes=0)

# Hadamard gate for the two qubits
HH = tf.tensordot(Hsingle, Hsingle, axes=0)

# Hadamard gate for the first qubit
IH = tf.tensordot(Isingle, Hsingle, axes=0)

# Hadamard gate for the second qubit
HI = tf.tensordot(Hsingle, Isingle, axes=0)

# ## $X_0\otimes X_1$ gate as a tensor product
XX = tf.tensordot(Xsingle, Xsingle, axes=0)

# Operator $X_0\otimes I_1$ ( X on first qubit)
XI = tf.tensordot(Xsingle, Isingle, axes=0)

# Operator $I_0\otimes X_1$ (X on the second qbuit)
IX = tf.tensordot(Isingle, Xsingle, axes=0)

# ## $Y_0\otimes Y_1$ gate as a tensor product
YY = tf.tensordot(Ysingle, Ysingle, axes=0)

# Operator $Y_0\otimes I_1$ ( Y on first qubit)
YI = tf.tensordot(Ysingle, Isingle, axes=0)

# Operator $I_0\otimes Y_1$ (Y on the second qbuit)
IY = tf.tensordot(Isingle, Ysingle, axes=0)

# ## $Z_0\otimes Z_1$ gate as a tensor product
ZZ = tf.tensordot(Zsingle, Zsingle, axes=0)

# Operator $Z_0\otimes I_1$ ( Z on first qubit)
ZI = tf.tensordot(Zsingle, Isingle, axes=0)

# Operator $I_0\otimes Z_1$ (Z on the second qbuit)
IZ = tf.tensordot(Isingle, Zsingle, axes=0)

# CNOT gate
CNOT_np = np.zeros((2, 2, 2, 2), dtype=npcomplex)
CNOT_np[0, 0, 0, 0] = 1.0
CNOT_np[0, 0, 1, 1] = 1.0
CNOT_np[1, 1, 0, 1] = 1.0
CNOT_np[1, 1, 1, 0] = 1.0
CNOT = tf.constant(CNOT_np, dtype=tfcomplex)

# CZ gate
CZ_np = np.zeros((2, 2, 2, 2), dtype=npcomplex)
CZ_np[0, 0, 0, 0] = 1
CZ_np[0, 0, 1, 1] = 1
CZ_np[1, 1, 0, 0] = 1
CZ_np[1, 1, 1, 1] = -1
CZ = tf.constant(CZ_np, dtype=tfcomplex)

# gate functions

# apply gate function on single qubit
@tf.function
def Gate1(A, psi):
    # here A is (2,2) tensor for a 2x2 matrix
    # psi is (2,) tensor for a single qubit
    # the gate is the matrix vector dot multiplication
    # as a contraction of index 1 for A
    # and index 0 for psi
    return tf.tensordot(A, psi, axes=[1, 0])


# apply gate function for two qubits
@tf.function
def Gate2(A, psi):
    return tf.tensordot(A, psi, axes=[[1, 3], [0, 1]])


# apply gate function for two qubits (alias)
@tf.function
def Gate(A, psi):
    return tf.tensordot(A, psi, axes=[[1, 3], [0, 1]])


# apply gate function for N bits
@tf.function
def GateNN(A, psi, nqubits):
    """Generalized gate function for an arbitrary number of qubits

    Params
    ------
    A : gate tensor
    psi : input tensor
    nqubits : number of qubits = size(psi)//2

    nqubits is needed for speed

    """
    covariant = [num for num in range(nqubits * 2) if num % 2 == 1]
    contravariant = [num for num in range(nqubits)]
    return tf.tensordot(A, psi, axes=[covariant, contravariant])


# Scalar product of Two Quit states
@tf.function
def Scalar2(a, b):
    # old version (better)
    a1 = tf.reshape(a, (4, 1))
    b1 = tf.reshape(b, (4, 1))
    sc = tf.tensordot(
        tf.transpose(a1, conjugate=True), b1, axes=1
    )  # must be a tensor not a scalar
    # Reshape does not cust in the proper way, need
    # to transpose some column first
    # remind the following gives erros as returns a scalar and not a tensor
    # sc = tf.tensordot(tf.transpose(a, conjugate=True),
    #                  b, axes=[[1, 0], [0, 1]])
    return sc


# Scalar product of a generic qubit (return a scalar)
@tf.function
def Scalar(a, b):
    # old version (better)
    a1 = tf.reshape(a, (tf.size(a), 1))
    b1 = tf.reshape(b, (tf.size(b), 1))
    sc = tf.tensordot(
        tf.transpose(a1, conjugate=True), b1, axes=1
    )  # must be a tensor not a scalar
    return sc


@tf.function
def ScalarOld(a, b):
    # old version (better)
    a1 = tf.reshape(a, (4, 1))
    b1 = tf.reshape(b, (4, 1))
    sc = tf.tensordot(
        tf.transpose(a1, conjugate=True), b1, axes=1
    )  # must be a tensor not a scalar
    return sc


@tf.function
def Transform2Matrix(A):
    """reshape a 2 qubit gate operator as 4x4
    WARNING: TODO, check this function
    """
    return tf.reshape(tf.transpose(A, perm=[1, 2, 0, 3]), (4, 4))


# redefine the operators as functions
@tf.function
def EX(theta):
    """return exp(1j theta X)"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EXS = ct * Isingle + 1j * st * Xsingle
    return EXS


@tf.function
def EY(theta):
    """return exp(1j theta Y)"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EYS = ct * Isingle + 1j * st * Ysingle
    return EYS


@tf.function
def EZ(theta):
    """return exp(1j theta Z)"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EZS = ct * Isingle + 1j * st * Zsingle
    return EZS


# redefine the operators as functions


@tf.function
def EZI(theta):
    """return exp(theta Z0) I"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EZS = ct * Isingle + 1j * st * Zsingle
    EZ0 = tf.tensordot(EZS, Isingle, axes=0)
    return EZ0


@tf.function
def EXI(theta):
    """return exp(theta X0) I"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EXS = ct * Isingle + 1j * st * Xsingle
    EX0 = tf.tensordot(EXS, Isingle, axes=0)
    return EX0


@tf.function
def EYI(theta):
    """return exp(theta Y0) I"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EYS = ct * Isingle + 1j * st * Ysingle
    EY0 = tf.tensordot(EYS, Isingle, axes=0)
    return EY0


@tf.function
def EIZ(theta):
    """return I exp(theta Z1)"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EZS = ct * Isingle + 1j * st * Zsingle
    EZ0 = tf.tensordot(Isingle, EZS, axes=0)
    return EZ0


@tf.function
def EIX(theta):
    """return I exp(theta X1)"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EXS = ct * Isingle + 1j * st * Xsingle
    EX0 = tf.tensordot(Isingle, EXS, axes=0)
    return EX0


@tf.function
def EIY(theta):
    """return I exp(theta Y1)"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EYS = ct * Isingle + 1j * st * Ysingle
    EY0 = tf.tensordot(Isingle, EYS, axes=0)
    return EY0


@tf.function
def EZZ(theta):
    """return exp(theta Z0 Z1)"""
    ct = tf.cast(tf.math.cos(theta), dtype=tfcomplex)
    st = tf.cast(tf.math.sin(theta), dtype=tfcomplex)
    EZZ = ct * II + 1j * st * ZZ
    return EZZ


# Von Neumann Entanglement Entropy
@tf.function
def VonNeumannEntropy2(psi):
    """Return the entanglement entropy for a two-qubit
    with tensor (2,2) psi
    Return the entropy and
    the squared principla values

    Example
    -------
    entropy, pr=VonNeurmannEntropy(psi)


    Params
    ------
    psi two-qubit tensor (2,2)

    Returns
    -------
    entropy real scalar with entropy
    pr squared principal values (2,)

    """
    d = tf.linalg.svd(psi, compute_uv=False)
    d2 = tf.abs(d) ** 2
    logd2 = tf.math.log(d2 + 1e-12) / tf.math.log(2.0)
    return tf.reduce_sum(-d2 * logd2), d2


# N-qubit tensors
