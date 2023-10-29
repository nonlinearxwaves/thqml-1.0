ThQML: Thinking Quantum Machine learning
========================================

[C. Conti](https://github.com/nonlinearxwaves)

Code for the book Quantum Machine Learning, https://doi.org/10.1007/978-3-031-44226-1

Being free of any licensing fees, ThQML is ideal for exploring quantum machine learning for students and researchers.

Created by following https://packaging.python.org/en/latest/tutorials/packaging-projects/

Requirements
------------
`graphviz` https://graphviz.gitlab.io/download for plot_model to work

Installation
------------
Local installation.
The use may eventually use a `conda` or similar environment.

In the downloaded folder from github
```bash
python3 -m pip install .
```
Note the dot at the end


To install an editable version

```bash
python3 -m pip install -e .
```
Note the dot after editable

Citing ThQML 
------------
If you use ThQML in your research, please cite the book
See the references in `thqml.bib`

Last test with code versions (september 2023)
---------------------------------------------
  - tensorflow 2.11.0
  - matplotlib 3.7.2
  - numpy 1.24.3

Tree
----
```
.
├── CODE_OF_CONDUCT.md
├── jupyternotebooks
│   ├── bosonsampling
│   │   ├── BosonSamplingExample1.ipynb
│   │   ├── BosonSamplingExample2.ipynb
│   │   ├── BosonSamplingExample3.ipynb
│   │   ├── BosonSamplingExample4b.ipynb
│   │   ├── BosonSamplingExample4.ipynb
│   │   ├── BosonSamplingExample5.ipynb
│   │   ├── BosonSamplingExample6.ipynb
│   │   ├── BosonSamplingExample7.ipynb
│   │   ├── BosonSamplingExample8.ipynb
│   │   └── BosonSamplingExample9.ipynb
│   ├── logo_circular.png
│   ├── phasespace
│   │   ├── beamsplitter.ipynb
│   │   ├── BellBS.ipynb
│   │   ├── coherentcomplex.ipynb
│   │   ├── coherentcomplextrainingCOV.ipynb
│   │   ├── coherentcomplextrainingDER.ipynb
│   │   ├── coherentcomplextraining.ipynb
│   │   ├── coherent.ipynb
│   │   ├── differentiallayer.ipynb
│   │   ├── phasemodulator.ipynb
│   │   ├── photoncountinglayer.ipynb
│   │   ├── singlemodesqueezerBS.ipynb
│   │   ├── singlemodesqueezer.ipynb
│   │   ├── symplectic.ipynb
│   │   ├── testGaussianLayer.ipynb
│   │   ├── twolayersreservoir.ipynb
│   │   ├── twomodesqueezer.ipynb
│   │   └── uncertainty.ipynb
│   ├── quantumfeaturemap
│   │   ├── coherentstate.ipynb
│   │   ├── kernelexample.ipynb
│   │   ├── QAOATwoQubitTransverseFieldIsing.ipynb
│   │   ├── QuantumKernelMachineQubits.ipynb
│   │   ├── QuantumNeuralStateTwoQubitTransverseFieldIsing.ipynb
│   │   ├── QubitsDensityMatrix.ipynb
│   │   ├── QubitsGym.ipynb
│   │   ├── QubitsMap.ipynb
│   │   ├── SingleQubitTransverseFieldIsing.ipynb
│   │   ├── squeezedvacuum.ipynb
│   │   ├── TensorsAndVectors.ipynb
│   │   └── TwoQubitTransverseFieldIsing.ipynb
│   └── soliton
│       ├── BoseHubbardNNT.ipynb
│       ├── BoseHubbardTwinNNT.ipynb
│       ├── BoseHubbardTwinNNTVersusN.ipynb
│       ├── modelSingleSoliton.png
│       └── modelTwin.png
├── LICENSE.txt
├── mathematica
│   ├── noncommutative.nb
│   ├── SingleQubitTransverseIsing.nb
│   ├── SingleQubitTransverseIsing.pdf
│   ├── tensorgaussian.nb
│   ├── TwoQubitTransverseIsing.nb
│   └── TwoQubitTransverseIsing.pdf
├── matlabsymbolic
│   ├── beamsplitter.m
│   ├── entanglementBS.m
│   ├── plot_entanglement_BS.m
│   ├── RqRpJ.m
│   ├── squeezedoperator.m
│   ├── test_RqRp.m
│   └── twomodesqueezedoperator.m
├── pyproject.toml
├── README.md
├── thqml
│   ├── EntangledFeatureMap.py
│   ├── __init__.py
│   ├── phasespace.py
│   ├── quantummap.py
│   ├── quantumsolitons.py
│   └── utilities.py
└── thqml.bib
```
Documentation
-------------

*See the book Quantum Machine Learning*

Code per chapter
----------------

<a id="org3b82add"></a>

### Chapter 1

-   `jupyternotebooks/quantumfeaturemap/kernelexample.ipynb`
      Page 13
    Example of generation of dataset and with `scikit-lean`


<a id="org622ba77"></a>

### Chapter 2

-   `jupyternotebooks/quantumfeaturemap/coherentstate.ipynb`
    Page 33
    Feature mapping by coherent state
-   `jupyternotebooks/quantumfeaturemap/squeezedvacuum.ipynb`
    page 38
    feature mapping by squeezed state


<a id="org916ab06"></a>

### Chapter 3

-   `jupyternotebooks/quantumfeaturemap/QubitsMap.ipynb`
    page 52
    defining Qubits in TensorFlow
-   `jupyternotebooks/quantumfeaturemap/TensorsAndVectors.ipynb`
    page 56
    defining and manipulating tensors and vectors
-   `thqml/quantummap.py`
    page 61
    main library with qubit functions
-   `jupyternotebooks/quantumfeaturemap/QubitsMap.ipynb`
    page 61 (continue)
    qubits feature mapping
-   `jupyternotebooks/quantumfeaturemap/QubitsGym.ipynb`
    page 64
    more on qubits and tensors
-   `thqml/quantummap.py`
    page 68 (continue)
    main library with qubit functions
-   `jupyternotebooks/quantumfeaturemap/QubitsMap.ipynb`
    page 74 (continue)
    qubits feature mapping
-   `jupyternotebooks/quantumfeaturemap/QuantumKernelMachineQubits.ipynb`
    page 79
    quantum kernel machine with qubits


<a id="orgdad239c"></a>

### Chapter 4

-   `jupyternotebooks/quantumfeaturemap/SingleQubitTransverseFieldIsing.ipynb`
    page 89
    Transverse Field Ising Model with a Single Qubit
-   `mathematica/SingleQubitTransverseIsing.nb` **MATHEMATICA**
    page 90
    Analytical results on the single qubit transverse field Ising model
-   `jupyternotebooks/quantumfeaturemap/SingleQubitTransverseFieldIsing.ipynb`
    page 92 (continue)
    Transverse Field Ising Model with a Single Qubit
-   `jupyternotebooks/quantumfeaturemap/SingleQubitTransverseFieldIsing.ipynb`
    page 99 (continue)
    Transverse Field Ising Model with a Single Qubit


<a id="org2fd7484"></a>

### Chapter 5

-   `mathematica/TwoQubitTransverseIsing.nb` **MATHEMATICA**
    page 104
    Analytical results on the two-qubit transverse field Ising model
-   `mathematica/TwoQubitTransverseIsing.nb` **MATHEMATICA**
    page 105 (continue)
    Analytical results on the two-qubit transverse field Ising model
-   `jupyternotebooks/quantumfeaturemap/QubitsDensityMatrix.ipynb`
    page 114
    Computing the density matrix with qubits
-   `jupyternotebooks/quantumfeaturemap/QubitsDensityMatrix.ipynb`
    page 118 (continue)
    Computing the density matrix with qubits
-   `jupyternotebooks/quantumfeaturemap/QubitsDensityMatrix.ipynb`
    page 132 (continue)
    Computing the density matrix with qubits
-   `mathematica/TwoQubitTransverseIsing.nb` **MATHEMATICA**
    page 133 (continue)
    Analytical results on the two-qubit transverse field Ising model


<a id="org0d23f64"></a>

### Chapter 6

-   `jupyternotebooks/quantumfeaturemap/TwoQubitTransverseFieldIsing.ipynb`
    page 139
    Transverse Field Ising Model with Two Qubit
-   `jupyternotebooks/quantumfeaturemap/QAOATwoQubitTransverseFieldIsing.ipynb`
    page 152
    Transverse Field Ising Model with Two Qubit with Quantum Approximation Optimization Algorithm
-   `jupyternotebooks/quantumfeaturemap/QuantumNeuralStateTwoQubitTransverseFieldIsing.ipynb`
    page 157
    Transverse Field Ising Model with Two Qubit with Quantum Neural State


<a id="org411d827"></a>

### Chapter 7

-   `matlabsymbolic/test_Rqpm.m`  **MATLAB**
    page 188
    Symbolic relations of projection matrices
-   `jupyternotebooks/phasespace/symplectic.ipynb`
    page 189
    Test of projection matrices in **jupyter**
-   `thqml/phasespace.py`
    page 190
    main library with phasespace functions
-   `thqml/phasespace.py`
    page 191 (continue)
    main library with phasespace functions


<a id="org3a2f4a9"></a>

### Chapter 8

-   `thqml/phasespace.py`
    page 196 (continue)
    main library with phasespace functions
-   `thqml/phasespace.py`
    page 199 (continue)
    main library with phasespace functions
-   `jupyternotebooks/phasespace/testGaussianLayer.ipynb`
    page 199
    Test of the Gaussian layer
-   `jupyternotebooks/phasespace/testGaussianLayer.ipynb`
    page 200 (continue)
    Test of the Gaussian layer
-   `thqml/phasespace.py`
    page 205 (continue)
    main library with phasespace functions
-   `jupyternotebooks/phasespace/coherent.ipynb`
    page 209
    Neural network representation of a coherent state
-   `thqml/phasespace.py`
    page 210 (continue)
    main library with phasespace functions


<a id="org92c5f7c"></a>

### Chapter 9

-   `jupyternotebooks/phasespace/coherent.ipynb`
    page 216 (continue)
    Neural network representation of a coherent state
-   `jupyternotebooks/phasespace/coherentcomplex.ipynb`
    page 217
    Coherent state in a complex medium
-   `jupyternotebooks/phasespace/coherentcomplex.ipynb`
    page 219 (continue)
    Coherent state in a complex medium
-   `jupyternotebooks/phasespace/coherentcomplextraining.ipynb`
    page 219
    Coherent state in a complex medium with training
-   `jupyternotebooks/phasespace/coherentcomplextraining.ipynb`
    page 221 (continue)
    Coherent state in a complex medium with training
-   `jupyternotebooks/phasespace/coherentcomplextraining.ipynb`
    page 222 (continue)
    Coherent state in a complex medium with training
-   `jupyternotebooks/phasespace/coherentcomplextrainingDER.ipynb`
    page 225
    Coherent state in a complex medium with training with derivatives
-   `thqml/phasespace.py`
    page 226 (continue)
    main library with phasespace functions
-   `jupyternotebooks/phasespace/coherentcomplextrainingCOV.ipynb`
    page 227
    Coherent state in a complex medium with training with covariance
-   `jupyternotebooks/phasespace/twolayersreservoir.ipynb`
    page 230
    Two trainable interferometer and a reservoir
-   `thqml/phasespace.py`
    page 231 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/phasemodulator.ipynb`
    page 232
    Phase modulator model


<a id="orgc275731"></a>

### Chapter 10

-   `matlabsymbolic/squeezeoperator.m` **MATLAB**
    page 237
    Matrix representation of the squeeze operator in MATLAB
-   `jupyternotebooks/phasespace/singlemodesquezer.ipynb`
    page 237
    Single-mode squeezer model
-   `thqml/phasespace.py`
    page 238 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/singlemodesquezer.ipynb`
    page 240 (continue)
    Single-mode squeezer model
-   `jupyternotebooks/phasespace/singlemodesquezer.ipynb`
    page 243 (continue)
    Single-mode squeezer model
-   `matlabsymbolic/squeezeoperator.m` **MATLAB**
    page 244
    Matrix representation of the two-mode squeeze operator in MATLAB
-   `thqml/phasespace.py`
    page 245 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/twomodesquezer.ipynb`
    page 247
    Two-mode squeezer model
-   `matlabsymbolic/beamsplitter.m` **MATLAB**
    page 248
    Matrix representation of the beam splitter operator in MATLAB
-   `thqml/phasespace.py`
    page 248 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/beamsplitter.ipynb`
    page 251
    Beam splitter model
-   `thqml/phasespace.py`
    page 251 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/photoncountinglayer.ipynb`
    page 252
    Example 1 with a photon counting layer
-   `jupyternotebooks/phasespace/BellBS.ipynb`
    page 254
    Example 2 with a photon counting layer
-   `jupyternotebooks/phasespace/photoncounting.ipynb`
    page 254
    Example 3 with a photon counting layer
-   `jupyternotebooks/phasespace/BellBS.ipynb`
    page 255 (continue)
    Example 4 with a photon counting layer and beam splitter


<a id="org1b09d13"></a>

### Chapter 11

-   `thqml/phasespace.py`
    page 263 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/uncertainty.ipynb`
    page 264
    Example in using Laplacian layer for computing uncertainty
-   `thqml/phasespace.py`
    page 265 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/uncertainty.ipynb`
    page 267 (continue)
    Example in using Laplacian layer for computing uncertainty
-   `thqml/phasespace.py`
    page 267 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/uncertainty.ipynb`
    page 269 (continue)
    Example in using Laplacian layer for computing uncertainty
-   `jupyternotebooks/phasespace/uncertainty.ipynb`
    page 272 (continue)
    Example in using Laplacian layer for computing uncertainty
-   `mathematica/noncommutative.nb` **MATHEMATICA**
    page 275
    Mathematica example on non commutative operators
-   `mathematica/tensorgaussian.nb`  **MATHEMATICA**
    page 276
    Mathematica example on tensors for Gaussian states
-   `thqml/phasespace.py`
    page 279 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/phasespace/differentiallayer.ipynb`
    page 282
    Example of use of differential layer in computing uncertainty
-   `jupyternotebooks/phasespace/BellBS.ipynb`
    page 284 (continue)
    Example 4 with a photon counting layer and beam splitter
-   `jupyternotebooks/phasespace/singlemodesqueezerBS.ipynb`
    page 287
    Model with single mode squeezer and beam splitter with entanglement
-   `thqml/phasespace.py`
    page 287 (continue)
    Main library with phasespace functions


<a id="org1602151"></a>

### Chapter 12

-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample1.ipynb`
    page 305
    Example 1 with boson sampling
-   `thqml/phasespace.py`
    page 309 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample1.ipynb`
    page 310
    Example 1 with boson sampling, GBS on single model coherent state
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample2.ipynb`
    page 312
    Example 2 with boson sampling, GBS on single mode squeezed state
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample3.ipynb`
    page 314
    Example 3 with boson sampling, GBS on multi-mode (two modes) coherent states
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample4.ipynb`
    page 316
    Example 4 with boson sampling, GBS on multi-mode (two modes) squezed and coherent states with transform layer
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample4b.ipynb`
    page 316
    Example 4b with boson sampling, GBS on multi-mode (two modes) squezed and coherent states with random layer
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample5.ipynb`
    page 318
    Example 5 with boson sampling, GBS Haar interferometer and multimode squeezed vacuum
-   `thqml/phasespace.py`
    page 319 (continue)
    Main library with phasespace functions
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample6.ipynb`
    page 321
    Example 6 with boson sampling, GBS Haar interferometer and multimode squeezed vacuum
    -   Generates the following figures
        1.  BosonSamplingExample6.pdf
        2.  modelHaar.pdf
        3.  BosonSamplingExample6ALL.pdf
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample6.ipynb`
    page 324 (continue)
    Example 6 with boson sampling, GBS Haar interferometer and multimode squeezed vacuum
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample6.ipynb`
    page 324 (continue)
    Example 6 with boson sampling, GBS Haar interferometer and multimode squeezed vacuum
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample7.ipynb`
    page 329
    Example 7 with boson sampling, GBS Haar and squeezer with training particle number
    -   Generates the following figures
        1.  modelBS7.png
        2.  modelBS7.pdf
        3.  BS7pairsnotraining.pdf
        4.  BS7quaternotraining.pdf
        5.  BS7ALLnotraining.pdf
        6.  BS7traininghistory.pdf
        7.  BS7pairstrained.pdf
        8.  BS7quatertrained.pdf
        9.  BS7ALLtrained.pdf
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample8.ipynb`
    page 336
    Example 8 with boson sampling, GBS Haar and squeezing training particle number and squeezing parameters
-   `jupyternotebooks/bosonsamplingexample/BosonSamplingExample9.ipynb`
    page 336
    Example 9 with boson sampling, GBS Haar and squeezing training on differential particle number


<a id="org963eb6e"></a>

### Chapter 13

-   `thqml/quantumsolitons.py`
    page 348
    Library with functions for quantum manybody and solitons
-   `thqml/quantumsolitons.py`
    page 354 (continue)
    Library with functions for quantum manybody and solitons
-   `jupyternotebooks/soliton/BoseHubbardNNT.ipynb`
    page 356
    Model for the ground state of the Bose-Hubbard Hamiltonian
-   `thqml/quantumsolitons.py`
    page 361 (continue)
    Library with functions for quantum manybody and solitons
-   `jupyternotebooks/soliton/BoseHubbardTwinNNT.ipynb`
    page 362
    Model Bose-Hubbard Hamiltonian with two solitons
-   `jupyternotebooks/soliton/BoseHubbardTwinNNTVersusN.ipynb`
    page 362
    Model Bose-Hubbard Hamiltonian with two solitons versus N


<a id="org2b058a2"></a>
