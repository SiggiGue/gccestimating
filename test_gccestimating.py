import pytest
import numpy as np

from pytest import assume

from gccestimating import GCC, corrlags
from gccestimating import _get_fftfuncs


def test_estimate():
    est = GCC.Estimate(
        name='test', 
        sig=np.random.randn(11), 
        spec=np.random.randn(5))
    assume(len(est) == 11)
    assume(np.all(np.asarray(est) == est.sig))
    assume(np.all(np.asarray(est, dtype=np.float16)==est.sig.astype(np.float16)))
    assume(est.index_to_lag(5) == 0)
    assume(est.index_to_lag(6, samplerate=2) == 1/2)


def test_gcc_autocorrelation():
    sig = np.random.randn(1024)
    gcc = GCC(sig, sig)
    cc = gcc.cc()
    assume(len(cc.sig) == len(sig)*2-1)
    assume(np.allclose(cc.sig[1023], np.sum(sig**2)))
    assume(np.asarray(cc, dtype=np.float16).dtype==np.float16)    
    assume(np.all(np.asarray(cc) ==cc.sig))    
    ccroth = gcc.roth()
    assume(ccroth.sig[1023]==1)
    assume(np.sum(ccroth.sig)==1)
    ccscot = gcc.scot()
    assume(ccscot.sig[1023]==1)
    assume(np.sum(ccscot.sig)==1)
    ccphat = gcc.phat()
    assume(ccphat.sig[1023]==1)
    assume(np.sum(ccphat.sig)==1)
    ccht = gcc.ht()
    assume(np.argmax(ccht.sig)==1023)
    coh = gcc.coherence()
    assume(np.allclose(coh, 1+0j))


def test_gcc_crosscorrelation():
    sig1 = np.random.randn(1024)
    sig2 = np.random.randn(1024)
    gcc = GCC(sig1, sig2)
    cc = gcc.cc()
    assume(len(cc.sig) == (len(sig1) + len(sig2) - 1))
    assume(np.allclose(cc.sig[1023], np.sum(sig1*sig2)))
    
    sig2 = np.random.randn(1024*2)
    gcc = GCC(sig1, sig2)
    cc = gcc.cc()
    assume(len(cc.sig) == (len(sig1) + len(sig2) - 1))


def test_gcc_from_spectra():
    sig1 = np.random.randn(1024)
    sig2 = np.random.randn(1024)
    spec1 = np.fft.rfft(sig1, 2*1024)
    spec2 = np.fft.rfft(sig2, 2*1024)
    gcc = GCC.from_spectra(spec1, spec2)
    cc = gcc.cc()
    assume(len(cc.sig) == (len(sig1) + len(sig2) - 1))
    assume(np.allclose(cc.sig[1023], np.sum(sig1*sig2)))
    assume(np.allclose(cc.sig, GCC(sig1, sig2).cc().sig))

    spec2 = np.fft.rfft(sig2, 1024)
    with pytest.raises(ValueError):
        gcc = GCC.from_spectra(spec1, spec2)

    spec1 = np.fft.fft(sig1, 2*1024)
    spec2 = np.fft.fft(sig2, 2*1024)
    gcc = GCC.from_spectra(spec1, spec2, onesided=False)
    cc = gcc.cc()
    assume(len(cc.sig) == (len(sig1) + len(sig2) - 1))
    assume(np.allclose(cc.sig[1023], np.sum(sig1*sig2)))
    assume(np.allclose(cc.sig, GCC(sig1, sig2).cc().sig))
    noise1 = np.random.randn(1024)*1e-3
    noise2 = np.random.randn(1024)*1e-3
    gcc = GCC(sig1+noise1, sig1+noise2)
    # eckart is only implement but tests are far away from complete:
    cceckart = gcc.eckart(sig1, noise1, noise2)


def test_corrlags():
    lags = corrlags(5)
    assume(list(lags)==[-2, -1, 0, 1, 2])
    lags = corrlags(5, 4)
    assume(list(lags)==[i/4 for i in [-2, -1, 0, 1, 2]])


def test__get_fftfuncs():
    x = np.random.randn(10) + 1j*np.random.randn(10)
    func1, func2 = _get_fftfuncs(x.real, x, x.imag)
    assume(func1 == np.fft.fft)
    assume(func2 == np.fft.ifft)

    func1, func2 = _get_fftfuncs(np.real(x))
    assume(func1 == np.fft.rfft)
    assume(func2 == np.fft.irfft)