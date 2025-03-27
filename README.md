pypeaks
=======

Identifying peaks from data is one of the most common tasks in many
research and development tasks. **pypeaks** is a python module to detect
peaks from any data like histograms and time-series.

Following are the available methods implemented in this module for peak detection:
* Slope based method, where peaks are located based on how the data varies.
* Intervals based method, where a set of intervals can be passed to provide apriori
information that there will be at most one peak in each interval, and we just pick the
maximum in each interval, filtering out irrelevant peaks at the end.
* A hybrid method which combines these two methods.

##Installation
```bash
$ sudo pip install --upgrade pypeaks
```

##Usage
There is an example case included along with the code. If you don't have this folder, please
load your data instead. Or get it from 
[https://github.com/gopalkoduri/pypeaks](https://github.com/gopalkoduri/pypeaks).

###Important note
The peak finding function expects a normalized smoothed histogram. It does smoothing by default.
If you want to change the smoothness, customize the corresponding argument. If the data
is not normalized (so that the area under the curve comes to 1), there is a function
provided to do that. If you don't get any peaks, then you probably overlooked this!


```python
import pickle
from pypeaks import Data, Intervals

[x, y] = pickle.load(file('examples/sample-histogram.pickle'))
data_obj = Data(x, y, smoothness=11)

#Peaks by slope method
data_obj.get_peaks(method='slope')
#print data_obj.peaks
data_obj.plot()

#Peaks by interval method
ji_intervals = pickle.load('examples/ji_intervals.pickle')
ji_intervals = Intervals(ji_intervals)
data_obj.get_peaks(method='interval', intervals=ji_intervals)
#print data_obj.peaks
data_obj.plot(intervals=ji_intervals)

#Read the help on Data object, and everything else is explained there.
help(Data)
```

In case you face some issue, report it on [github](https://github.com/gopalkoduri/pypeaks),
or write to me at **gopala [dot] koduri [at] gmail [dot] com**!
