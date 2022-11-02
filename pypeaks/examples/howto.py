# -*- coding: utf-8 -*-

#Import necessary modules

import pickle
from pypeaks import Data, Intervals

#Load data
#We already have provided some data samples and a script in examples/ directory. 
#If you don't have it, you can either load your own data, or download them from 
#https://github.com/gopalkoduri/pypeaks

data = pickle.load(open("../examples/sample-histogram.pickle"))
hist = Data(data[0], data[1])

#Get peaks by slope method and plot them

hist.get_peaks(method='slope')
hist.plot()

#Get peaks by interval method and plot them

#In the example/ folder, there is a pickle file with some example intervals, 
#in this case, just-intonation intervals for music. They can refer to any intervals!
ji_intervals = pickle.load(open('../examples/ji-intervals.pickle'))
ji_intervals = Intervals(ji_intervals)

hist.get_peaks(method='interval', intervals=ji_intervals)
hist.plot(intervals=ji_intervals.intervals)

#Accessing the peaks data
#The Data object has x, y_raw, y, smoothness and peaks variables available. 
#The help functions shows the methods available for it:
#try help(Data) in a python/ipython interpreter

