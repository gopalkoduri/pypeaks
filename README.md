pypeaks
=======

##Introduction
Python program which has a number of ways to detect peaks from any data. 
This particular implementation is primarily focussed on histogram data 
obtained from pitch values of an audio music recording. It has a number 
of functions which have been well documented. The most useful function is
peaks() function which takes a number of arguments besides the data.

There are mainly the following methods implemented for peak detection:
* Slope based method
* Peaks based on just intonation intervals (Or any custom set of intervals)
* Peaks based on equi-tempered intervals (with custom interval width)
* A hybrid method which combines the best of the three methods

##Usage
```python
import pickle
import matplotlib.plot as plt
from scipy.ndimage.filters import gaussian_filter

[x, y] = pickle.load(file('data/sample-histogram.pickle'))
y_smoothed = gaussian_filter(y, 11)
plot(x, ys)
```
