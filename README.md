setFTs
======
This library provides functionalities for calculating the Fourier transform on set functions, 
based on the novel mathematical foundation of discrete signal processing on set functions [1].
We provide functionalities for:
- Initializing a set function object from a full set of function evaluations.
- Inititalizing a set function object from a queryable python function.
- Applying the fast Fourier transform algorithm [1]
- Applying sparse Fourier transform algorithm [2]
- Set function minimization algorithms [3]
- Shapley Values Calculation
## 1 Documentation
Full documentation of the setfunctions and plotting modules can be found at: https://ebners.github.io/setFTs_docs/
or in the Documentation_setFTs.pdf provided in the repositors

## 2 Requirements
setFTs uses the python library pySCIPOpt for the implementation of the MIP-based minimization algorithm. 
pySCIPOpt requires a working installation of the SCIP Optimization Suite. The creators of pySCIPOpt recommend using conda as it installs SCIP automatically. And allows the installation of pySCIPOpt in one command:
```
conda install --channel conda-forge pyscipopt
```
More information about installing pySCIPOpt can be found at: https://github.com/scipopt/PySCIPOpt/blob/master/INSTALL.md

## 3 Installation
The installation of our package works over pypi and therefore a working installation of pip is needed. The pip command to install setFTs is the following:
```
pip install setFTs
```
## References
[1]
```
@article{Discrete_Signal_Proc, 
     title={Discrete signal processing with set functions},
     volume={69},
     DOI={10.1109/tsp.2020.3046972},
     journal={IEEE Transactions on Signal Processing},
     author={Puschel, Markus and Wendler, Chris},
     year={2021}, 
     pages={1039–1053}
 } 
```
 [2]
```
 @article{Sparse,
    author    = {Chris Wendler and
               Andisheh Amrollahi and
               Bastian Seifert and
               Andreas Krause and
               Markus P{\"{u}}schel},
    title     = {Learning Set Functions that are Sparse in Non-Orthogonal Fourier Bases},
    journal   = {CoRR},
    volume    = {abs/2010.00439},
    year      = {2020},
    url       = {https://arxiv.org/abs/2010.00439},
}
```
[3]
```
@article{MIPS,
    author={Weissteiner,Jakob and Wendler, Chris and Seuken, Sven and Lubin,Ben and Püschel, Markus},
    title={Fourier analysis-based iterative combinatorial auctions},
    DOI={10.24963/ijcai.2022/78},
    journal={Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence},
    year={2022}
    } 
```
