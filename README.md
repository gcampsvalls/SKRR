# SKRR - Deep Structured Kernel Ridge Regression

This is a repo for the Structured Kernel Ridge Regression (SKRR) method presented in "Retrieval of Physical Parameters with 
Deep Structured Kernel Regression" by Gustau Camps-Valls et al.

<b>Abstract.</b> Retrieval of physical parameters is of paramount relevance for Earth monitoring. Statistical (machine) learning approaches have been successfully introduced in the community because they can learn nonlinear functional relations from observational data with no strong {\em a priori} assumptions and parametric forms. However, these methods still have two relevant problems: they only consider only one nonlinear feature map of the data, which can be limiting in complex problems where inputs (e.g. radiances) and outputs (e.g. state vectors) have strong nonlinear relations, and in most of the cases models do not incorporate the structure of the output (dependent) variables. This paper proposes a kernel method that solves the two aforementioned problems for physical parameter retrieval: first, it performs multioutput regression with the desired number of connected mappings, and secondly, it incorporates the output variables structure via a dedicated kernel. The proposed method has a closed-form solution and thus neither kernel dimensionality reduction nor pre-imaging is necessary unlike in previous structured kernel methods. Through the definition of appropriate kernel feature mappings, we also derive a pragmatic deep structured kernel ridge regression. The method is characterized statistically using a Gaussian process treatment and providing guarantees based on the concepts of leverage scores and effective dimension: Both explain that including output structure acts as a powerful regularizer. We illustrate the method's performance in toy examples and remote sensing parameter estimation problems involving vegetation parameters (chlorophyll, LAI, and fractional vegetation cover) from CHRIS images and the atmospheric temperature, moisture, and ozone profiles from IASI data.

## What you will find here:

- A basic implementation in MATLAB of the algorithm: 
    - [Simple code snippets and demos](https://github.com/IPL-UV/SKRR/code)
    - [Full ISP SimpleR Matlab Toolbox](https://github.com/IPL-UV/simpleR)
- Additional results. 

## Cite our work: 

You may cite the paper with the following bibitem:
```
@article {Camps-Vall22skrr,
  author = {Camps-Valls, Gustau and Campos-Taberner, Manuel and aparra, Valero and Martino, Luca and Mu\~noz-Mar\'i, Jordi},
  title = {Retrieval of Physical Parameters with 
Deep Structured Kernel Regression},
  volume = {},
  number = {},
  year = {2022},
  doi = {},
  publisher = {},
  journal = {}
}
```
