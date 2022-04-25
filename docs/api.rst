APIs
====

Introduction
------------

.. code-block:: python

   import soundfile as sf
   import scipy.signal as ss
   from pybss.bss.iva import NaturalGradLaplaceIVA

   waveform_mix, _ = sf.read("sample-2ch.wav")
   print(waveform_mix.shape) # (160000, 2)
   iva = NaturalGradLaplaceIVA()
   _, _, spectrogram_mix = ss.stft(waveform_mix.T, nperseg=4096, noverlap=2048)
   spectrogram_est = iva(spectrogram_mix)
   print(spectrogram_mix.shape, spectrogram_est.shape) # (2, 2049, 80)

Submodules
----------

.. toctree::
   :maxdepth: 1

   pybss.algorithm
   pybss.bss