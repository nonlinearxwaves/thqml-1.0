#!/usr/bin/env python
# coding: utf-8

# # Example of quantum classifier with 2 qubits
#
#
#
# <img src="../img/logo_circular.png" width="20" height="20" />@by claudio<br>
# nonlinearxwaves@gmail.com<br>
#
#
# @created 21 jan 2022

# In[1]:


from matplotlib import cm
import matplotlib.pyplot as plt
from quomplex2021.utilities import utilities
import numpy as np
import tensorflow as tf
import os
ROOT_DIR = os.path.dirname(os.path.abspath(
    os.path.dirname(os.path.abspath(os.curdir))))
os.chdir(ROOT_DIR)
print("Current working directory: {0}".format(os.getcwd()))


# In[2]:


# In[3]:


mytype = tf.complex64


# ## Define the qbuit as an ancilla state

# In[4]:


qubit0 = tf.constant([1, 0], dtype=mytype)
print(qubit0)


# In[5]:


psi = tf.tensordot(qubit0, qubit0, axes=0)
print(psi)


# Define $|0\rangle \otimes |0 \rangle$

# In[6]:


q00 = tf.tensordot(qubit0, qubit0, axes=0)


# In[7]:


q00


# # Define the Hadamard Gate

# ## Hadamard Gate for the single qubit

# In[8]:


Hsingle = tf.constant([[1/np.sqrt(2), 1/np.sqrt(2)],
                      [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=mytype)
print(Hsingle)


# Test the Hadamard gate on $|0\rangle$

# In[9]:


tf.tensordot(Hsingle, qubit0, axes=1)


# ## Hadamard Gate for two qubits as an outer tensor product

# In[10]:


H = tf.tensordot(Hsingle, Hsingle, axes=0)
print(H)


# ## Apply the Hadamard gate to $|00\rangle$

# In[11]:


tf.tensordot(H, q00, axes=[[1, 3], [0, 1]]),


# # Define a function for applying a gate

# In[12]:


@tf.function
def gate(A, psi):
    return tf.tensordot(A, psi, axes=[[1, 3], [0, 1]])


# ### Apply the H gate on the $|00\rangle$ and store the resulting state

# In[13]:


Hq00 = gate(H, q00)


# In[14]:


print(Hq00)


# # Define the Z gate

# ## Single Q-bit identity

# In[15]:


Isingle = tf.constant([[1, 0], [0, 1]], dtype=mytype)


# ## Single Q-bit Z gate

# In[16]:


Zsingle = tf.constant([[1, 0], [0, -1]], dtype=mytype)
print(Zsingle)


# ## $Z_0\otimes Z_1$ gate as a tensor product

# In[17]:


ZZ = tf.tensordot(Zsingle, Zsingle, axes=0)
print(ZZ)


# ## Operator $Z_0\otimes I_1$ ( Z on first qubit)

# In[18]:


Z0 = tf.tensordot(Zsingle, Isingle, axes=0)
print(Z0)


# ## Operator $I_0\otimes Z_1$ (Z on the second qbuit)

# In[19]:


Z1 = tf.tensordot(Isingle, Zsingle, axes=0)
print(Z1)


# **Remark the Z0 and Z1 are different ! the outer tensor product takes into account order in the order of the indices !**

# ## Test Z-gate

# In[20]:


gate(Z0, q00)


# In[21]:


gate(Z1, q00)


# In[22]:


gate(ZZ, q00)


# In[23]:


gate(ZZ, Hq00)


# In[24]:


gate(Z0, Hq00)


# In[25]:


gate(Z1, Hq00)


# # Operator exponent of Z

# ## Exponent of Z in a single Q bit  $e^{\imath \theta_z Z}$

# In[26]:


Zsingle_complex = tf.constant(1j*Zsingle.numpy(), dtype=tf.complex64)
print(Zsingle_complex)


# In[27]:


thetaZ = tf.Variable(1.1, dtype=tf.float32)


# ### Use the exponent of matrix function

# In[28]:


tf.linalg.expm(1j*tf.cast(thetaZ, dtype=tf.complex64)
               * tf.cast(Zsingle, dtype=tf.complex64))


# ### Use the identity $e^{\imath \theta_Z Z}=cos(\theta_Z)I+\imath sin(\theta_Z)Z$

# In[29]:


EthetaZsingle = tf.cast(tf.math.cos(thetaZ), dtype=tf.complex64) * \
                        Isingle+1j*tf.cast(tf.math.sin(thetaZ),
                                           dtype=tf.complex64)*Zsingle


# In[30]:


print(EthetaZsingle)


# ## Exponent of Z0 only $e^{\imath \theta_Z Z}\otimes I$

# In[31]:


expZ0 = tf.tensordot(EthetaZsingle, Isingle, axes=0)


# ### Test gate

# In[32]:


gate(expZ0, q00)


# In[33]:


gate(expZ0, Hq00)


# ## Exponent of Z1 only $I\otimes e^{\imath \theta_Z Z}I$

# In[34]:


expZ1 = tf.tensordot(Isingle, EthetaZsingle, axes=0)


# ### Test gate

# In[35]:


gate(expZ1, q00)


# In[36]:


gate(expZ1, Hq00)


# # Define X gate

# ## X gate for the single qbit

# In[37]:


Xsingle = tf.constant([[0, 1], [1, 0]], dtype=mytype)
print(Zsingle)


# ## Exponent of the X gate

# In[38]:


thetaX = tf.Variable(0.3, dtype=tf.float32)


# In[39]:


EthetaXsingle = tf.cast(tf.math.cos(thetaX), dtype=tf.complex64) * \
                        Isingle+1j*tf.cast(tf.math.sin(thetaX),
                                           dtype=tf.complex64)*Xsingle


# ## Exponent of X0 only $e^{\imath \theta_X X}\otimes I$

# In[40]:


expX0 = tf.tensordot(EthetaXsingle, Isingle, axes=0)


# ## Exponent of X1 only $I\otimes e^{\imath \theta_X X}$

# In[41]:


expX1 = tf.tensordot(Isingle, EthetaXsingle, axes=0)


# # Define Y gate

# ## Y gate for the single qbit

# In[42]:


Y_np = np.array([[0.0, complex(0, -1)], [complex(0, 1), 0.0]],
                dtype=np.complex64)


# In[43]:


Y_np


# In[44]:


Ysingle = tf.constant(Y_np, dtype=mytype)
print(Ysingle)


# ## Exponent of the Y gate

# In[45]:


thetaY = tf.Variable(-0.7, dtype=tf.float32)


# In[46]:


EthetaYsingle = tf.cast(tf.math.cos(thetaY), dtype=tf.complex64) * \
                        Isingle+1j*tf.cast(tf.math.sin(thetaY),
                                           dtype=tf.complex64)*Ysingle


# ## Exponent of Y0 only $e^{\imath \theta_Y Y}\otimes I$

# In[47]:


expY0 = tf.tensordot(EthetaYsingle, Isingle, axes=0)


# ## Exponent of X1 only $I\otimes e^{\imath \theta_Y Y}$

# In[48]:


expY1 = tf.tensordot(Isingle, EthetaYsingle, axes=0)


# # CNOT gate

# In[49]:


CNOT_np = np.zeros((2, 2, 2, 2))
CNOT_np[0, 0, 0, 0] = 1
CNOT_np[0, 0, 1, 1] = 1
CNOT_np[1, 1, 0, 1] = 1
CNOT_np[1, 1, 1, 0] = 1


# In[50]:


CNOT = tf.constant(CNOT_np, dtype=tf.complex64)


# In[51]:


CNOT


# In[52]:


tf.transpose(CNOT, perm=[1, 3, 0, 2])


# In[53]:


print(tf.reshape(CNOT, (4, 4)))


# In[54]:


print(tf.reshape(tf.transpose(CNOT, perm=[1, 3, 0, 2]), (4, 4)))


# Reshape does not cust in the proper way, need to transpose some column first

# In[55]:


@tf.function
def transform(A):
    """ reshape an operator as 4x4 """
    return (tf.reshape(tf.transpose(A, perm=[1, 3, 0, 2]), (4, 4)))


# In[56]:


print(transform(CNOT))


# ## Test CNOT

# In[57]:


gate(CNOT, q00)


# In[58]:


gate(H, Hq00)


# In[59]:


gate(ZZ, Hq00)


# In[60]:


gate(CNOT, gate(ZZ, Hq00))


# In[61]:


gate(CNOT, gate(ZZ, Hq00))


# # CZ gate

# In[62]:


CZ_np = np.zeros((2, 2, 2, 2))
CZ_np[0, 0, 0, 0] = 1
CZ_np[0, 0, 1, 1] = 1
CZ_np[1, 1, 0, 0] = 1
CZ_np[1, 1, 1, 1] = -1


# In[63]:


CZ = tf.constant(CZ_np, dtype=tf.complex64)


# In[64]:


print(transform(CZ))


# ## Exponent of ZZ $e^{\imath \theta_{01} Z_0\otimes Z_1}$

# In[65]:


II = tf.tensordot(Isingle, Isingle, axes=0)
II


# In[66]:


theta01 = tf.Variable(1.1, dtype=tf.float32)


# In[67]:


expZ0Z1 = tf.cast(tf.math.cos(theta01), dtype=mytype)*II+1j * \
                  ZZ*tf.cast(tf.math.sin(theta01), dtype=mytype)


# In[68]:


print(transform(expZ0Z1))


# In[69]:


utilities.printonscreen(transform(expZ0Z1))


# # Feature MAP

# ## redefine the operators as functions

# In[70]:


@tf.function
def expZ0(theta):
    """ return exp(theta Z0) I """
    tc = tf.cast(theta, dtype=tf.complex64)
    EZS = tf.math.cos(tc)*Isingle+1j*tf.math.sin(tc)*Zsingle
    EZ0 = tf.tensordot(EZS, Isingle, axes=0)
    return EZ0


@tf.function
def expX0(theta):
    """ return exp(theta X0) I """
    tc = tf.cast(theta, dtype=tf.complex64)
    EXS = tf.math.cos(tc)*Isingle+1j*tf.math.sin(tc)*Xsingle
    EX0 = tf.tensordot(EXS, Isingle, axes=0)
    return EX0


@tf.function
def expY0(theta):
    """ return exp(theta Y0) I """
    tc = tf.cast(theta, dtype=tf.complex64)
    EYS = tf.math.cos(tc)*Isingle+1j*tf.math.sin(tc)*Ysingle
    EY0 = tf.tensordot(EYS, Isingle, axes=0)
    return EY0


@tf.function
def expZ1(theta):
    """ return I exp(theta Z1) """
    tc = tf.cast(theta, dtype=tf.complex64)
    EZS = tf.math.cos(tc)*Isingle+1j*tf.math.sin(tc)*Zsingle
    EZ0 = tf.tensordot(Isingle, EZS, axes=0)
    return EZ0


@tf.function
def expX1(theta):
    """ return I exp(theta X1) """
    tc = tf.cast(theta, dtype=tf.complex64)
    EXS = tf.math.cos(tc)*Isingle+1j*tf.math.sin(tc)*Xsingle
    EX0 = tf.tensordot(Isingle, EXS, axes=0)
    return EX0


@tf.function
def expY1(theta):
    """ return I exp(theta Y1) """
    tc = tf.cast(theta, dtype=tf.complex64)
    EYS = tf.math.cos(tc)*Isingle+1j*tf.math.sin(tc)*Ysingle
    EY0 = tf.tensordot(Isingle, EYS, axes=0)
    return EY0


@tf.function
def expZZ(theta):
    """ return exp(theta Z0 Z1) """
    tc = tf.cast(theta, dtype=tf.complex64)
    EZZ = tf.math.cos(tc)*II+1j*ZZ*tf.math.sin(tc)
    return EZZ


# ### test the function

# In[71]:


gate(expZ0(np.pi), Hq00)


# In[72]:


gate(expZZ(1.1), Hq00)


# In[73]:


gate(expZ0Z1, Hq00)


# In[74]:


# Feature map


# In[75]:


@tf.function
def featuremapU(phi, theta0, theta1, theta01):
    phi1 = gate(H, phi)
    phi2 = gate(expZ0(theta0), phi1)
    phi3 = gate(expZ1(theta1), phi2)
    phi4 = gate(expZZ(theta01), phi3)
    return phi4


# In[76]:


Upsi = featuremapU(psi, 1.0, 2.0, -1.0)


# # Scalar product

# In[77]:


@tf.function
def scalarproduct(a, b):
    a1 = tf.reshape(a, (4, 1))
    b1 = tf.reshape(b, (4, 1))
    return tf.tensordot(tf.transpose(a1, conjugate=True), b1, axes=1)


# In[78]:

scalarproduct(gate(H, psi), gate(H, psi))


# # Project on the 00 state

# In[79]:


scalarproduct(q00, Upsi)


# In[80]:


@tf.function
def probability00(psi):
    sp = scalarproduct(q00, psi)
    return tf.pow(tf.abs(sp), 2)


# In[81]:


tf.print(probability00(psi))


# In[82]:


tf.print(probability00(Upsi))


# In[83]:


# CZ gate


# # Variational circuit

# In[84]:


@tf.function
def VariationalW1(psi, tX0, tY0, tX1, tY1):
    phi = gate(expX0(tX0), psi)
    phi = gate(expY0(tY0), phi)
    phi = gate(expX1(tX1), phi)
    phi = gate(expY1(tY1), phi)
    phi = gate(CZ, phi)
    return phi


# Test 1 variational layer

# In[85]:


Wpsi = VariationalW1(Hq00, 0.2, -1.5, 2.1, 1.0)
print(Wpsi)


# In[86]:


np.random.random(4)


# Test 3 variational layers (with random parameters)

# In[87]:


phi = Hq00
for j in range(3):
    params = np.random.random(4)
    print('Layer '+repr(j)+' Parameters '+repr(params))
    phi = VariationalW1(phi, params[0], params[1], params[2], params[3])
print(phi)


# # Build the model

# ## First define dedicated layer for feature map

# In[88]:


class FeatureMapLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(FeatureMapLayer, self).__init__()

    def call(self, inputs):
        """ inputs are
        """
        psi = inputs[0]
        thetaZ0 = inputs[1]
        thetaZ1 = inputs[2]
        thetaZ01 = inputs[3]

        psio = featuremapU(psi, thetaZ0, thetaZ1, thetaZ01)
        return psio


# In[89]:


class VariationalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(VariationalLayer, self).__init__()
        # here t[0]=tZ0 t[1]=tY0 t[2]=tZ1 t[3]=tY1
        t_np = np.random.random(7)
        self.t = tf.Variable(t_np, dtype=tf.float32)

    def call(self, inputs):
        """ inputs are
        """
        phi = inputs
        phi = gate(expZ0(self.t[0]), phi)
        phi = gate(expY0(self.t[1]), phi)
        phi = gate(expX0(self.t[2]), phi)
        phi = gate(expZ1(self.t[3]), phi)
        phi = gate(expY1(self.t[4]), phi)
        phi = gate(expX1(self.t[5]), phi)
        phi = gate(CZ, phi)
        phi = gate(expZZ(self.t[6]), phi)
        phi = gate(CNOT, phi)
        return phi


# In[90]:


class MeasurementLayer(tf.keras.layers.Layer):
    """ this depends on the f(z) function """

    def __init__(self, y):
        super(MeasurementLayer, self).__init__()
        self.y = tf.constant(y, dtype=tf.complex64)

    def call(self, inputs):
        """ inputs are
        """
        phi = inputs
        # here f(z)=1
        # phi=gate(0.5*(II+self.y*ZZ), phi)
        phi = gate(ZZ, phi)
        return phi


# In[91]:


class ScalarProductLayer(tf.keras.layers.Layer):
    """ this depends on the f(z) function """

    def __init__(self):
        super(ScalarProductLayer, self).__init__()

    def call(self, inputs):
        """ inputs are
        """
        a = inputs[0]
        b = inputs[1]
        sc = scalarproduct(a, b)
        return sc


# ## Define the model

# In[92]:


depth = 4


# In[93]:


x = tf.keras.Input(2, dtype=tf.float32)


# In[94]:


theta0 = x[0]
theta1 = x[1]
theta01 = (np.pi-theta0)*(np.pi-theta1)/((np.pi+theta0)*(np.pi+theta1))
# theta01 = (np.pi-theta0)*(np.pi-theta1)


# In[95]:


phi = FeatureMapLayer()([q00, theta0, theta1, theta01])


# In[96]:


phi = FeatureMapLayer()([phi, theta0, theta1, theta01])


# In[97]:


Wphi = phi
for j in range(depth):
    Wphi = VariationalLayer()(Wphi)


# In[98]:


M1p = MeasurementLayer(1.0)(Wphi)


# In[99]:


M1m = MeasurementLayer(-1.0)(Wphi)


# In[100]:


meanM1p = tf.math.real(ScalarProductLayer()([Wphi, M1p]))


# In[101]:


meanM1m = tf.math.real(ScalarProductLayer()([Wphi, M1m]))


# In[102]:


# sig = tf.sigmoid(meanM1p-meanM1m)
sig = meanM1p


# In[103]:


# meanM=tf.keras.activations.hard_sigmoid(meanM+2.0)


# In[104]:


model = tf.keras.Model(inputs=x, outputs=sig)


# In[105]:


Nplot = 30
xmin = 0.5
xmax = 1.5
x0 = np.linspace(xmin, xmax, Nplot)
x1 = np.linspace(xmin, xmax, Nplot)
xin = np.zeros(2,)


# In[106]:


# model.summary()


# # Plot the function with no training

# In[107]:


surf0 = np.zeros((len(x0), len(x1)))
for i0 in range(len(x0)):
    xin[0] = x0[i0]
    for i1 in range(len(x1)):
        xin[1] = x1[i1]
        surf0[i0, i1] = model(xin)


# In[108]:


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
[Xp, Yp] = np.meshgrid(x0, x1)
surf = ax.plot_surface(Xp, Yp, surf0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig, ax = plt.subplots()
ax.pcolor(Xp, Yp, surf0)


# # Define a fitting function

# In[109]:


def target_function(x, y):
#    f=np.power(np.sin(x/2)*np.sin(y/2),2.0)
    f = np.sin(y)
    return f


# In[110]:


Nbatch = 1000
ytrain = np.zeros(Nbatch)
xtrain = xmin+(xmax-xmin)*np.random.random((Nbatch, 2))
for i0 in range(Nbatch):
    ytrain[i0] = target_function(xtrain[i0, 0], xtrain[i0, 1])


# In[111]:


max(xtrain[:, 1])


# ## Plot target function

# In[112]:


surfT = np.zeros((len(x0), len(x1)))
for i0 in range(len(x0)):
    xin[0] = x0[i0]
    for i1 in range(len(x1)):
        xin[1] = x1[i1]
        surfT[i0, i1] = target_function(xin[0], xin[1])


# In[113]:


fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax[0].plot_surface(Xp, Yp, surf0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf = ax[1].plot_surface(Xp, Yp, surfT, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig, ax = plt.subplots(1, 2)
ax[0].pcolor(Xp, Yp, surf0)
ax[1].pcolor(Xp, Yp, surfT)


# # Train

# In[114]:


model.trainable_weights


# In[115]:


model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='adam')


# In[116]:


history = model.fit(xtrain, ytrain, epochs=200, verbose=0)
print('Final loss '+repr(history.history['loss'][-1]))
plt.plot(history.history['loss'])


# # Plot model after training

# In[117]:


surfF = np.zeros((len(x0), len(x1)))
for i0 in range(len(x0)):
    xin[0] = x0[i0]
    for i1 in range(len(x1)):
        xin[1] = x1[i1]
        surfF[i0, i1] = model(xin)


# In[118]:


fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
# Plot the surface.
[Xp, Yp] = np.meshgrid(x0, x1)
surf = ax[0].plot_surface(Xp, Yp, surf0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf = ax[1].plot_surface(Xp, Yp, surfF, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf = ax[2].plot_surface(Xp, Yp, surfT, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig, axs = plt.subplots(1, 3)
axs[0].pcolor(Xp, Yp, surf0)
axs[1].pcolor(Xp, Yp, surfF)
axs[2].pcolor(Xp, Yp, surfT)


# In[119]:


print(ciao


# # Generate the dataset

# In[ ]:


from sklearn.datasets import make_moons, make_circles, make_classification


# In[ ]:


datasets=make_moons(noise=0.0)


# In[ ]:


xtrain, ytrain=datasets


# In[ ]:


X, y=make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng=np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable=(X, y)


# In[ ]:


xtrain, ytrain=linearly_separable


# In[ ]:


len(ytrain)


# In[ ]:


Nbatch=100
xtrain=2.0*np.pi*np.random.random((Nbatch, 2))


# In[ ]:


# xtrain


# In[ ]:


ytrain=np.zeros(Nbatch)
for j in range(len(xtrain)):
    if xtrain[j, 0] > np.pi:
        ytrain[j]=1.0


# In[ ]:


ytrain


# In[ ]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


# In[ ]:


model.weights


# In[ ]:


history=model.fit(xtrain, ytrain, epochs=1000, verbose=0)


# In[ ]:


model.weights


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['binary_accuracy'])


# In[ ]:


history.history.keys()


# In[ ]:


ytrain


# In[ ]:


xtrain.shape


# In[ ]:


out=np.zeros(len(xtrain))
for j in range(len(xtrain)):
    out[j]=model(xtrain[j, :])


# In[ ]:


tf.round(out)


# In[ ]:


class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ScaleLayer, self).__init__()
      self.scale=tf.Variable(1.0, dtype=tf.float32)
    def call(self, inputs):
      expZZalpha=tf.cast(tf.math.cos(self.scale), dtype=tf.complex64) * \
                         II+1j*Z*tf.cast(tf.math.sin(self.scale),
                                         dtype=tf.complex64)
      return tf.tensordot(expZZalpha, inputs, axes=[[1, 3], [0, 1]])


# In[ ]:


class GateLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(GateLayer, self).__init__()
    def call(self, inputs):
      return tf.tensordot(inputs[0], inputs[1], axes=[[1, 3], [0, 1]])


# In[ ]:


class ScalarProductLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ScalarProductLayer, self).__init__()
    def call(self, inputs):
        a=inputs[0]
        b=inputs[1]
        a1=tf.reshape(a, (4, 1))
        b1=tf.reshape(b, (4, 1))
        return tf.tensordot(tf.transpose(a1, conjugate=True), b1, axes=1)


# In[ ]:


# expZZalpha = tf.complex(tf.math.cos(alpha),0.0)*II+1j*Z*tf.complex(tf.math.sin(alpha),0.0)


# In[ ]:


# SL=ScaleLayer()


# In[ ]:


Hq00=gate(H, q00)


# In[ ]:


psiA=GateLayer()([expZ0theta, Hq00])
psiB=GateLayer()([expZ1theta, psiA])
psiU=GateLayer()([expZZtheta, psiB])


# In[ ]:


# Hq00 = tf.tensordot(H,q00, axes=[[1,3],[0,1]])


# In[ ]:


# psiA =tf.tensordot(expZ0theta,Hq00, axes=[[1,3],[0,1]])


# In[ ]:


# psiB =tf.tensordot(expZ1theta,Hq00, axes=[[1,3],[0,1]])


# In[ ]:


# psiU =tf.tensordot(expZZtheta,psiA, axes=[[1,3],[0,1]])


# In[ ]:


# psiV=SL(psiU)
psiV=ScaleLayer()(psiU)


# In[ ]:


# a1 = tf.reshape(psiU,(4,1))
# b1 = tf.reshape(psiV, (4,1))
# out11=tf.pow(tf.abs(tf.tensordot(tf.transpose(a1,conjugate=True),b1,axes=1)),2)
# out11=tf.pow(tf.abs(ScalarProductLayer()([psiU,psiV])),2.0)


# In[ ]:


# psiV =tf.tensordot(expZZalpha,psiU, axes=[[1,3],[0,1]])


# In[ ]:


# UL=tf.keras.layers.Lambda(U)


# In[ ]:


# pL=tf.keras.layers.Lambda(probability00)


# In[ ]:


# psiU=UL(psi1)


# In[ ]:


# out00=pL(psiV)


# In[ ]:


# model =tf.keras.Model(inputs=x, outputs=[out11, out00, psiV])


# In[ ]:


# model.add_loss(tf.exp(out11))


# In[ ]:


# model.compile()


# In[ ]:


xin=np.zeros(2)


# In[ ]:


xin.shape


# In[ ]:


xin[0]=3.2


# In[ ]:


xin


# In[ ]:


model(xin)


# In[ ]:


xin[1]=2.0


# In[ ]:


model(xin)


# In[ ]:


model.summary()


# In[ ]:


Nbatch=10
xtrain=np.random.rand(Nbatch, 2)-0.5


# In[ ]:


model.fit(xtrain, epochs=1000, verbose=0)


# In[ ]:


model.weights


# In[ ]:


o1=model(xtrain)


# In[ ]:


o1[0].numpy()


# In[ ]:


yout=np.zeros(Nbatch)
yout.shape


# In[ ]:


yout[0]


# In[ ]:


for j in range(Nbatch):
    yout[j]=model(xtrain[j, :])[0].numpy()[0][0]


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:



plt.plot(yout)


# In[ ]:


tf.keras.utils.plot_model(model, to_file='./jupyternotebooks/quantumfeaturemap/model_entangled.png',
                          show_shapes=False, show_dtype=False,
                            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=300,
                            layer_range=None, show_layer_activations=False)


# In[ ]:
