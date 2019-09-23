Micropsi Custom Loss Function
==============================

A custom loss function, that ought to be used to minimize loss value being produced by 
a predictive model. 

The predictive model returns K estimates of a target 3D vector. The aim of the implemented 
loss function is to minimize the distance of each estimated vector.

The goal is achieved through calculating an eucledian distance of each predicted vector to 
a target vector and summarizing the values. 

The function supports batch optimization, in which case it calculates loss value for every 
given target vector and returns a batch of loss values.


## Requirements
A tensorflow module is required to use this function.
