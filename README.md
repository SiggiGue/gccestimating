# Generalized Cross Correlation (GCC) Estimates

[![CodeFactor](https://www.codefactor.io/repository/github/siggigue/gccestimating/badge)](https://www.codefactor.io/repository/github/siggigue/gccestimating)

[![Coverage Status](https://coveralls.io/repos/github/SiggiGue/gccestimating/badge.svg)](https://coveralls.io/github/SiggiGue/gccestimating)

[![Documentation Status](https://readthedocs.org/projects/gccestimating/badge/?version=latest)](https://gccestimating.readthedocs.io/en/latest/?badge=latest)

This project provides estimators for the generalized cross correlation according to *Knapp and Carter 1976* [KC76].


## Implemented Estimators (compare [KC76])

The generalized Estimator can be described by


<img src="https://render.githubusercontent.com/render/math?math=\hat{R}_{xy}^{(\text{g})} = \int_{-\infty}^{\infty}{\psi_\text{g}(f) G_{xy}(f)~e^{\text{j} 2\pi f \tau} df}">

where <img src="https://render.githubusercontent.com/render/math?math=G_{xy}(f)"> denotes the cross power spectrum of <img src="https://render.githubusercontent.com/render/math?math=x(t)"> and <img src="https://render.githubusercontent.com/render/math?math=y(t)">.
In this project, all estimates are computed in the spectral domain using the *Wiener-Kinchin relations* (e.g. <img src="https://render.githubusercontent.com/render/math?math=G_{xx}=X(f)X^{*}(f)">).

Following estimators are implemented:

- **Cross Correlation** 
  <img src="https://render.githubusercontent.com/render/math?math=\psi_{\text{CC}}=1">
  

- **Roth**; same as the <img src="https://render.githubusercontent.com/render/math?math=`H_1`"> estimator describing the Wiener-Hopf filter
  <img src="https://render.githubusercontent.com/render/math?math=\psi_{\text{Roth}} = \frac{1}{G_{xx}(f)}">

- **Smoothed Coherence Transform** (SCOT): 
  <img src="https://render.githubusercontent.com/render/math?math=\psi_{\text{SCOT}} = \frac{1}{\sqrt{G_{xx}(f)G_{yy}(f)}}">

- **PHAse Transform** (PHAT): 
  <img src="https://render.githubusercontent.com/render/math?math=\psi_{\text{PHAT}} = \frac{1}{|G_{xy}(f)|}">

- Eckart
  <img src="https://render.githubusercontent.com/render/math?math=\psi_{\text{Eckart}} = \frac{G_{uu}(f)}{G_{nn}(f)G_{mm}(f)}">

- **Hanan Thomson** (HT), Maximum Likelihood  estimator 
  <img src="https://render.githubusercontent.com/render/math?math=\psi_{\text{HT}} = \psi_{\text{ML}} = \frac{\left|\gamma_{xy}(f)\right|^2}{\left|G_{xy}\right| \left(1-\gamma_{xy}(f)\right)^2}">
  with 
  <img src="https://render.githubusercontent.com/render/math?math=\gamma_{xy}(f) = \frac{G_{xy}(f)}{\sqrt{G_{xx}(f)G_{yy}(f)}}">

## Insalling

This repo uses a `pyproject.toml` file generated with the dependency and package managing tool *poetry*.

This package can be installed with an up to date pip
`pip install .`

or using poetry
`poetry install`

otherwise use your own favorite way to install/use the code in your environment.


## Example

```python
import numpy as np
import matplotlib.pylab as plt
from gccestimating import GCC, corrlags

 # generate some noise signals
nsamp = 1024

noise1 =  0.5*np.random.randn(nsamp)
sig1 = np.zeros(nsamp) + noise1

noise2 =  0.5*np.random.randn(nsamp)
sig2 = np.zeros_like(sig1) + noise2

noise_both = np.random.randn(256)

sig1[:256] = noise_both
sig2[500:756] = noise_both

# create a lags array
lags = corrlags(2*nsamp-1, samplerate=1)

# Create the a GCC instance    
gcc = GCC(sig1, sig2)

def mkplot(est, p):
    plt.subplot(p)
    plt.plot(lags, est.sig, label=est.name)
    plt.legend()

# calculate the standard cc estimate
cc_est = gcc.cc()

# plot it using the mkplot function
mkplot(cc_est, 611)

# plot the other estimates
mkplot(gcc.scot(), 612)
mkplot(gcc.phat(), 613)
mkplot(gcc.roth(), 614)
mkplot(gcc.ht(), 615)
mkplot(gcc.eckart(noise_both, noise1, noise2), 616)

# compare cc to the timedomain based 
# implementation from Numpy
# you will see: very close (errors < 1e-13)
plt.figure()
plt.plot(np.correlate(sig1, sig2, 'full'))
plt.plot(gcc.cc())
plt.show()

```

## References

[KC76]: Knapp and Carter, "The Generalized Correlation Method for Estimation of Time Delay", IEEE Trans. Acoust., Speech, Signal Processing, August, 1976
