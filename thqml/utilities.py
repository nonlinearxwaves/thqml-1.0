# %% class for useful output operations with tensorflow

import numpy as np

class utilities:
    """Various utilities."""

    @staticmethod
    def printonscreen(VT):
      #print a tensor a matrix on the screen with a given precision
      VTnp = VT.numpy()
      N = VTnp.shape[0]
      M = VTnp.shape[1]
      for ix in range(N):
        for iy in range(M):
          re = np.real(VTnp[ix,iy])
          im = np.imag(VTnp[ix,iy])
          print('{:+02.1f}{:+02.1f}i'.format(re,im),end=" ")
        print("") #print endlie

    @staticmethod
    def printonscreennp(VTnp):
      #print a tensor a matrix on the screen with a given precision
      N = VTnp.shape[0]
      M = VTnp.shape[1]
      for ix in range(N):
        for iy in range(M):
          re = np.real(VTnp[ix,iy])
          im = np.imag(VTnp[ix,iy])
          print('{:+02.1f}{:+02.1f}i'.format(re,im),end=" ")
        print("") #print endlie
