# Processing description of signals

-initial_signals:
    - rats: desc todo
    - monkey: desc todo
    - humans: desc todo
- proccessed signals:
    - bua: 
        1. Bandpass raw signals with a butter filter of order 4 and frequencies [300, 6000] Hz. 
        2. take the absolute value
        3. lowpass the resulting signal at 300Hz
        4. interpolate the result (to downsample) at 1000Hz (just to reduce data size)
    - neurons: generate a continuous signal at 1000Hz and counting the number of spiketimes in each bin. No smoothing.
- data is zscored
- fft computations on 1s hann windows, with 50% overlap, linear detrending
- welch: mean norm of all window values on time dimension
- coherence: on pairs of ffts by computing (fft1 * conj(fft2)).mean("time_dim") *I forgot to divide by welch !!!*
  
