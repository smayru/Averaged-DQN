# Averaged-DQN
chainer implementatio of [Averaged-DQN](http://proceedings.mlr.press/v70/anschel17a.html "Averaged-DQN").
This code is partly based on [here](http://ensekitt.hatenablog.com/entry/2016/11/28/035827).

## Abstract
By taking the average of the latst *k* parameters for estimaing the Q-function, Averaged-DQN stablizes the performance.
If *k* is *1*, this is essentially the same as standard DQN.

## How to use
python averaged_dqn.py --K=k --Episode=episode　　

## Analysis

I check the estimation error of Q-function varying the value of *k*.  

|*k*=1|*k*=2|*k*=3|*k*=5|*k*=10|
|:--:|:--:|:--:|:--:|:--:|
|53.98|10.27|1.43|1.42|0.69|

By increasing the value of *k*, you can reduce estimation error.  

Next, I checked the average reward for each episode.  

|*k*=1|*k*=2|*k*=3|*k*=5|*k*=10|
|:--:|:--:|:--:|:--:|:--:|
|152.36|151.85|149.69|165.04|130.29|  

When setting the value of *k* to be 5, it shows the best performance.

The detail is described in [averaged_dqn_analysis.ipynb](https://github.com/smayru/Averaged-DQN/blob/master/averaged_dqn_analysis.ipynb "averaged_dqn_analysis.ipynb").





