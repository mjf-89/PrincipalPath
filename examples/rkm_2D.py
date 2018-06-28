import numpy as np

from matplotlib import pyplot
from matplotlib.widgets import Slider, Button

import sklearn as skl
from sklearn import datasets

import linear_utilities as lu
import principalpath as pp

####
#PARAMETERS to be set by the user
####
    #seed for numpy.random
np.random.seed(7)

    #N, noise for toy dataset generation
N=1000
noise=0.2

    #Number of optimization variables i.e. path waypoints
NC=50

####
#generate/load dataset
####
    #S_curve
#[X,y] = skl.datasets.make_s_curve(N,noise)
#X = X[:,[0,2]]

    #Moons 
#[X,y] = skl.datasets.make_moons(N,noise=noise)

    #Constellation
X=np.genfromtxt('../datasets/2D_constellation.csv',delimiter=',')

    #Circle
#X=np.genfromtxt('../datasets/2D_circle.csv',delimiter=',')

    #Sin
#X=np.genfromtxt('../datasets/2D_sin.csv',delimiter=',')

    #Dumped sin
#X=np.genfromtxt('../datasets/2D_dmp.csv',delimiter=',')

    #Gaussians
#X=np.genfromtxt('../datasets/2D_gss.csv',delimiter=',')

N=X.shape[0]
d=X.shape[1]


####
#select boundaries
####
boundary_ids=lu.getMouseSamples2D(X,2)


####
#prefilter the data
####
#[fig,ax]=pyplot.subplots()
[X, boundary_ids, X_g]=pp.rkm_prefilter(X, boundary_ids, plot_ax=None)


####
#initialize waypoints
####
waypoint_ids = lu.initMedoids(X, NC, 'kpp',boundary_ids)
waypoint_ids = np.hstack([boundary_ids[0],waypoint_ids,boundary_ids[1]])
W_init = X[waypoint_ids,:] 


####
#annealing with rkm
####
#[fig,ax]=pyplot.subplots()
s_span=np.logspace(5,-5)
s_span=np.hstack([s_span,0])
models=np.ndarray([s_span.size,NC+2,d])
for i,s in enumerate(s_span):
    [W,u]=pp.rkm(X, W_init, s, plot_ax=None)
    W_init = W
    models[i,:,:] = W


####
#interactive model selection
####
s_id = pp.rkm_MS_gui(models,s_span,X,X_g)
print("You selected model with s={:.2f}".format(s_span[s_id]))

raw_input('press [enter] to quit')
