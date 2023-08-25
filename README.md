ThQML: Thinking Quantum Machine learning
========================================

[C. Conti](https://github.com/nonlinearxwaves)

Code for the book Quantum Machine Learning, thinking and exploration (ThMQL)

Being free of any licensing fees, ThQML is ideal for exploring quantum machine learning for students and researchers.

Created by following https://packaging.python.org/en/latest/tutorials/packaging-projects/

Installation
------------
Local installation.
The use may eventually use a =conda= or similar environment.

In the downloaded folder from github
```bash
python3 -m pip install -e .
```
Note the dot after editable

Citing ThQML 
------------
If you use ThQML in your research, please cite the book
See the references in `thqml.bib`

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

### Chapter 1
- jupyternotebooks/quantumfeaturemap/kernelexample.ipynb
  Example of generation of dataset and with =scikit-lean=
