"""Generalized Cross Correlation Estimators.

Istanciation Signatures:

- `gcc = GCC(sig1, sig2, fftlen)`
- `gcc = GCC.from_spectra(spec1, spec2, onesided=True)`

Estimators:

`gcc.cc()`, `gcc.roth()`, `gcc.scot()`, `gcc.phat()`, `gcc.ht()`

`gcc.gamma12()`

"""
from dataclasses import dataclass as _dataclass
import numpy as _np



class GCC(object):
    """Returns a GCC instance.

    Provides estimation methods for Generalized Cross Correlation.

    Parameters
    ----------
    sig1 : ndarray
        First signal.
    sig2 : ndarray 
        Second signal.
    fftlen : int or None
        Length of fft to be computed. 
        If None, it will be calculated automatically as next power of two.

    Returns
    -------
    gcc : GCC

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>> from gccestimating import GCC, corrlags

    >>> # generate some noise signals
    >>> nsamp = 1024
    
    >>> noise1 =  0.5*np.random.randn(nsamp)
    >>> sig1 = np.zeros(nsamp) + noise1
    
    >>> noise2 =  0.5*np.random.randn(nsamp)
    >>> sig2 = np.zeros_like(sig1) + noise2
    
    >>> noise_both = np.random.randn(256)
    
    >>> sig1[:256] = noise_both
    >>> sig2[500:756] = noise_both
    
    >>> # create a lags array
    >>> lags = corrlags(2*nsamp-1, samplerate=1)
    
    >>> # Create the a GCC instance    
    >>> gcc = GCC(sig1, sig2)
    
    >>> def mkplot(est, p):
    >>>     plt.subplot(p)
    >>>     plt.plot(lags, est.sig, label=est.name)
    >>>     plt.legend()
    
    >>> # calculate the standard cc estimate
    >>> cc_est = gcc.cc()
    
    >>> # plot it using the mkplot function
    >>> mkplot(cc_est, 611)
    
    >>> # plot the other estimates
    >>> mkplot(gcc.scot(), 612)
    >>> mkplot(gcc.phat(), 613)
    >>> mkplot(gcc.roth(), 614)
    >>> mkplot(gcc.ht(), 615)
    >>> mkplot(gcc.eckart(noise_both, noise1, noise2), 616)
    >>> plt.figure()
    >>> plt.plot(np.correlate(sig1, sig2, 'full'))
    >>> plt.plot(gcc.cc())
    >>> plt.show()

    """

    @classmethod
    def from_spectra(cls, spec1, spec2, onesided=True):
        """Returns a GCC instance.

        Parameters
        ----------
        spec1 : ndarray
            First spectrum.
        spec2 : ndarray 
            Second spectrum.
        onesided : bool
            If you provide twosided Spectra 
            (e.g. of comples signals) set to False.
            Default is True.

        Returns
        -------
        gcc : GCC

        """
        instance = cls()
        length1 = len(spec1)
        
        if len(spec2) != length1:
            raise ValueError('Spectra must be of same dimension.')
        
        instance._spec1 = spec1
        instance._spec2 = spec2

        if onesided:
            instance._fftlen = 2*length1-2 if length1%2 else 2*length1-1 
            instance._fft = _np.fft.rfft
            instance._ifft = _np.fft.irfft 
        else:
            instance._fftlen = length1
            instance._fft = _np.fft.fft
            instance._ifft = _np.fft.ifft 
        instance._corrlen = instance._fftlen - 1
        return instance

    def __init__(self, sig1=None, sig2=None, fftlen=None):
        """Returns a GCC instance.

        Parameters
        ----------
        sig1 : ndarray (N,)
            First signal.
        sig2 : ndarray (M,)
            Second signal.
        fftlen : int or None
            Length of fft to be computed. 
            Will be calculated automatically if None 
            (next power of two of 2max(N,M)-1).

        Returns
        -------
        gcc : GCC

        """
        self._sig1 = None
        self._sig2 = None
        self._spec11 = None
        self._spec22 = None
        self._spec12 = None
        self._gamma12 = None

        self._cc = None
        self._roth = None
        self._scot = None
        self._phat = None
        self._eckart = None
        self._ht = None

        self._corrlen = None
        self._fftlen = fftlen

        if sig1 is not None:
            self._from_signals(sig1, sig2, fftlen)

    def _from_signals(self, sig1, sig2, fftlen=None):
        corrlen = len(sig1) + len(sig2) - 1
        fftlen = fftlen or int(2**_np.ceil(_np.log2(corrlen)))
        fft, ifft = _get_fftfuncs(sig1, sig2)
        spec1 = fft(sig1, fftlen)
        spec2 = fft(sig2, fftlen)
        self._corrlen = corrlen
        self._fftlen = fftlen
        self._fft = fft  
        self._ifft = ifft
        self._sig1 = sig1
        self._sig2 = sig2   
        self._spec1 = spec1
        self._spec2 = spec2   

    def _backtransform(self, spec):
        sig = self._ifft(spec, self._fftlen)
        sig = _np.roll(sig, len(sig)//2)
        start = (len(sig)-self._corrlen)//2 + 1
        return sig[start:start+self._corrlen]

    @property
    def spec11(self):
        """Returns auto power spectrum of first signal."""
        if self._spec11 is None:
            self._spec11 = _np.real(self._spec1 * _np.conj(self._spec1))
        return self._spec11

    @property
    def spec22(self):
        """Returns auto power spectrum of second signal."""
        if self._spec22 is None:
            self._spec22 = _np.real(self._spec2 * _np.conj(self._spec2))
        return self._spec22

    @property
    def spec12(self):
        """Returns cross power spectrum of first and second signal."""
        if self._spec12 is None:
            self._spec12 = self._spec1*_np.conj(self._spec2)
        return self._spec12


    @_dataclass(init=True, repr=True, eq=True)
    class Estimate():
        """Data of an Estimate. 
        Instances are returned by estimators in GCC.
        
        Parameters
        ----------
        name : str
            Name of the estimator.
        sig : ndarray
            Estimator signal array (Rxy(t), Cross Correlation).
        spec : ndarray
            Estimator spectrum (Rxy(f)).

        """
        name: str
        sig: _np.ndarray
        spec: _np.ndarray

        def __array__(self, dtype=None):
            if dtype is not None:
                return self.sig.astype(dtype)
            else:
                return self.sig

        def __len__(self):
            return len(self.sig)

        def index_to_lag(self, index, samplerate=None):
            lag = (index - len(self.sig)//2) 
            if samplerate:
                lag /= samplerate
            return lag

    def cc(self):
        """Returns GCC estimate 
        
        $\\mathcal{F}^{-1} (S_{xy})$
        
        """         
        if self._cc is None:
            self._cc = GCC.Estimate(
                name='CC', 
                sig=self._backtransform(self.spec12), 
                spec=self.spec12)

        return self._cc

    def roth(self):
        """Returns GCC Roth estimate 
    
        $\\mathcal{F}^{-1} (S_{xy}/S_{xx})$

        """
        if self._roth is None:
            spec = self.spec12 / _prevent_zerodivision(self.spec11)
            self._roth = GCC.Estimate(
                name='Roth', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._roth

    def scot(self):
        """Returns GCC SCOT estimate 
        
        Smoothed gamma12 Transformed GCC.
    
        $\\mathcal{F}^{-1} (S_{xy}/\\sqrt{S_{xx}S_{yy}})$
        
        """        
        if self._scot is None:
            spec = self.gamma12()
            self._scot = GCC.Estimate(
                name='SCOT', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._scot
        
    def gamma12(self):
        """Returns gamma12 $\\gamma_{12}(f)$"""
        if self._gamma12 is None:
            self._gamma12 = self.spec12 / _prevent_zerodivision(
                _np.sqrt(self.spec11*self.spec22))
        return self._gamma12

    def coherence(self):
        """Returns the coherence."""
        return self.gamma12()**2

    def phat(self):
        """Returns GCC PHAT estimate 
        
        PHAse Transformed GCC.
        
        $\\mathcal{F}^{-1}(S_{xy}/|S_{xy}|)$
        
        """        
        if self._phat is None:
            spec = self.spec12 / _prevent_zerodivision(_np.abs(self.spec12))
            self._phat = GCC.Estimate(
                name='PHAT', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._phat

    def eckart(self, sig0, noise1, noise2):
        """Returns an eckart estimate.
        
        Parameters
        ----------
        sig0 : ndarray
            estimate of the actual signal to be correlated.
        noise1 : ndarray
            estimated noise in sig1.
        noise2 : ndarray
            estimated noise in sig2

        Returns
        -------
        estmate : GCC.Estimate

        Note
        ----
        only implemented, not fully tested.

        """
        if self._eckart is None:
            spec_sig0 = self._fft(sig0, self._fftlen)
            spec_noise1 = self._fft(noise1, self._fftlen)
            spec_noise2 = self._fft(noise2, self._fftlen)
            spec_sig00 = _np.real(spec_sig0*spec_sig0.conj())
            spec_noise11 = _np.real(spec_noise1*spec_noise1.conj())
            spec_noise22 = _np.real(spec_noise2*spec_noise2.conj())
            weight = spec_sig00 /_prevent_zerodivision(
                spec_noise11*spec_noise22)
            spec = self.spec12*weight
            self._eckart = GCC.Estimate(
                name='Eckart', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._eckart

    def ht(self):
        """Returns GCC HT estimate"""
        if self._ht is None:
            coh = _np.abs(self.gamma12())**2
            spec = self.spec12*coh/_prevent_zerodivision(_np.abs(self.spec12)*(1-coh))
            self._ht = GCC.Estimate(
                name='HT', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._ht


def corrlags(corrlen, samplerate=1):
    """Returns array of lags.
    
    Parameters
    ----------
    corrlen : int
        Lenght of correlation function (usually 2N-1).
    samplerate : scalar
    
    Returns
    -------
    lags : ndarray
    
    """
    dt = 1 / samplerate
    la = corrlen // 2
    lb = la+1 if corrlen%2 else la
    return _np.arange(-la*dt, lb*dt, dt)


def _get_fftfuncs(*signals):
    """Returns fft, ifft function depending on given signals data type."""
    if _np.all([_np.all(_np.isreal(sig)) for sig in signals]):
        return _np.fft.rfft, _np.fft.irfft
    else:
        return _np.fft.fft, _np.fft.ifft


def _prevent_zerodivision(sig, reg=1e-12, rep=1e-12):
    """Replaces values smaler reg. Same for negative values and negative reg.
    
    Replaces 
    
    `sig < reg & sig >= 0` with `rep`
    
    and
    
    `sig > -reg & sig <= 0` with `-rep`

    Parameters
    ----------
    sig : ndarray
        Will be modified by this function. 
        Provide a copy of your array if original is needed.
    reg : scalar
        All values around 0 (-reg, reg) are replaced by `rep`
    rep : scalar
        Replace value.

    Returns
    -------
    sig : ndarray
        Modified sig usable for division.

    """
    reg = abs(reg)
    rep = abs(rep)
    sig[_np.logical_and(sig < reg, sig >= 0)] = rep
    sig[_np.logical_and(sig > -reg, sig <= 0)] = -rep
    return sig
