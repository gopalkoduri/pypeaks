pypeaks
=======

##Introduction
Identifying peaks from some data is one of the most common tasks in many
research and development tasks. **pypeaks** is a python module which has 
a number of ways to detect peaks from any data, like histograms and time-series.

Following are the available methods implemented in this module for peak detection:
* Slope based method, where peaks are located based on how the data varies.
* Peaks based on intervals, where a set of intervals can be passed to provide appriori
information tha there will be at most one peak in each interval.
* A hybrid method which combines the best of these two methods.

###Important note
These functions expect a normalized smoothed histogram. It does smoothing by default.
If you want to change the smoothness, customize the corresponding argument.

##Usage
We have included an example case along with the data. If you dont have this folder, please
load your data instead. Or get it from 
[https://github.com/gopalkoduri/pypeaks](https://github.com/gopalkoduri/pypeaks).

```python
import pickle
from pypeaks.data import Data
from pypeaks.intervals import Intervals

[x, y] = pickle.load(file('examples/sample-histogram.pickle'))
data_obj = Data(x, y, smoothness=11)

\#Peaks by slope method
data_obj.get_peaks(method='slope')
\#print data_obj.peaks
data_obj.plot()

\#Peaks by interval method
ji_intervals = pickle.load('examples/ji_intervals.pickle')
ji_intervals = Intervals(ji_intervals)
data_obj.get_peaks(method='interval', intervals=ji_intervals)
\#print data_obj.peaks
data_obj.plot(intervals=ji_intervals)


```
