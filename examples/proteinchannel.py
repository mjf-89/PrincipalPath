import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import linear_utilities as lu
import principalpath as pp


####
#PARAMETERS to be set by the user
####
    #file name containing the grid
grid_file='../datasets/mol.status.txt'

    #number of waypoints for the path (excluded 2 boundaries)
NC = 20

    #downsampling factor (to alleviate burden on principal path finding)
ds = 10

    #s values for the principal path annealing 
s_span=np.logspace(6,-2)
s_span=np.hstack([s_span,0])

    #coordinate for boundary points
    #if they are not provided, they will be selected as centroids of the first and last z-slice of the channel
#boundary=np.ndarray([2,3],float)
#boundary[1,:] = [0.53,0.48,0.47];
#boundary[0,:] = [0.46,0.52,0.24];

    #whether or not to show the inflation 'animation'
inflate_animation=False


####
#sphere mesh, credit to http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut5.html
####
def sphere(C=np.zeros(3,float), r=1.0, n=25):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    x = x*r+C[0]
    y = y*r+C[1]
    z = z*r+C[2]

    return x, y, z


####
#sphere mask 
####
def sphere_mask(dlt, r_dlt):
    r = r_dlt * dlt
    x = np.arange(-r_dlt, r_dlt+1)*dlt
    mask = np.zeros([x.size,x.size,x.size],int)

    for i,xx in enumerate(x): 
        for j,yy in enumerate(x): 
            for k,zz in enumerate(x): 
                if(xx*xx+yy*yy+zz*zz<=r*r):
                    mask[i,j,k]=1

    return mask


####
#inflate and push sphere
####
def inflate(inout, C_id, dlt, ax_plot=None):
    inflating = 0
    r_dlt = 0
    while inflating<10:
        contacts=0
        while(contacts==0):
            if(ax_plot is not None):
                pyplot.ion()
                if(inflating):
                    ax_plot.collections.pop()

                [x,y,z]=sphere(np.asarray([C_id[1],C_id[0],C_id[2]])*dlt,r_dlt*dlt,10)
                ax_plot.plot_surface(x,y,z,color='b')
                pyplot.pause(1.0/60)

            C_ret = C_id.copy()*dlt
            C_ret = [C_ret[1],C_ret[0],C_ret[2]]
            r = r_dlt*dlt

            r_dlt = r_dlt+1
            mask = sphere_mask(dlt, r_dlt)
            inout_mask = inout[C_id[0]-r_dlt:C_id[0]+r_dlt+1, C_id[1]-r_dlt:C_id[1]+r_dlt+1, C_id[2]-r_dlt:C_id[2]+r_dlt+1] * mask

            contacts = np.sum(inout_mask)

        push = np.zeros(6,int)
        push[0] = np.sum(inout_mask[:r_dlt+2,:,:])
        push[1] = np.sum(inout_mask[r_dlt+1:,:,:])
        push[2] = np.sum(inout_mask[:,:r_dlt+2,:])
        push[3] = np.sum(inout_mask[:,r_dlt+1:,:])
        push[4] = np.sum(inout_mask[:,:,:r_dlt+2])
        push[5] = np.sum(inout_mask[:,:,r_dlt+1:])

        if(push[0]>0):
            C_id[0] = C_id[0]+1
        if(push[1]>0):
            C_id[0] = C_id[0]-1
        if(push[2]>0):
            C_id[1] = C_id[1]+1
        if(push[3]>0):
            C_id[1] = C_id[1]-1
        if(push[4]>0):
            C_id[2] = C_id[2]+1
        if(push[5]>0):
            C_id[2] = C_id[2]-1

        inflating = inflating+1
        r_dlt = r_dlt-1
    return C_ret, r
            

####
#load grid file (exported from NS) and prepare pp dataset
####
inout=np.genfromtxt(grid_file,dtype=int )
inout[inout<0]=0
inout[inout>0]=1
Nx = inout.shape[1]
dlt = 1.0/Nx
inout = np.reshape(inout, [Nx,Nx,Nx], 'F')

x = np.linspace(0,1,Nx)
[x,y,z] = np.meshgrid(x,x,x)

X = np.stack([x[inout==0],y[inout==0],z[inout==0]],1)
X = X[::ds,:]

N = X.shape[0]
d = X.shape[1]

####
#find principal path 
####
    #select boundary points if they weren't provided
if(not 'boundary' in locals()):
    boundary=np.ndarray([2,3],float)
    i=0
    while(np.all(inout[:,:,i]>0)):
        i = i+1
    boundary[0,0]=np.mean(x[inout[:,:,i]==0,i])
    boundary[0,1]=np.mean(y[inout[:,:,i]==0,i])
    boundary[0,2]=z[0,0,i]

    i=Nx-1
    while(np.all(inout[:,:,i]>0)):
        i = i-1
    boundary[1,0]=np.mean(x[inout[:,:,i]==0,i])
    boundary[1,1]=np.mean(y[inout[:,:,i]==0,i])
    boundary[1,2]=z[0,0,i]


    #initialize waypoints
waypoint_ids = lu.initMedoids(X, NC, 'kpp')
W_init = np.vstack([boundary[0,:], X[waypoint_ids,:], boundary[1,:]])

    #annealing with rkm
models=np.ndarray([s_span.size,NC+2,d])
for i,s in enumerate(s_span):
    [W,u]=pp.rkm(X, W_init, s, plot_ax=None)
    W_init = W
    models[i,:,:] = W

s_id = pp.rkm_MS_gui(models, s_span, X)
W = models[s_id,:,:]

####
#estimate radius (inflating spheres, the stupid way) 
####
    #inflate spheres centering them on the principal path

if(inflate_animation):
    lim=[np.min(X), np.max(X)]
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],alpha=0.5)
    ax.plot(W[:,0],W[:,1],W[:,2],'-ro')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
else:
    ax=None

W_=W.copy()
r = np.zeros(NC+2, float)
for i in range(1,NC+2):
    C_id = np.round(W[i,:]/dlt)
    C_id = np.asarray([C_id[1],C_id[0],C_id[2]])
    C_id = C_id.astype(int)
    W_[i,:], r[i] = inflate(inout, C_id, dlt, ax)

    #find linear coordinate for the corrected path
t=np.hstack([0,np.cumsum(np.diag(distance.cdist(W_,W_,'euclidean'),1))])

    #plot results
[fig,ax]=pyplot.subplots(2,2)
fig.suptitle('Orthogonal projection of Principal Path and Inflated Principal Path')
ax[0,0].scatter(X[:,0],X[:,2],alpha=0.5)
ax[0,0].plot(W[:,0],W[:,2],'-rx')
ax[0,0].plot(W_[:,0],W_[:,2],'-yx')
ax[0,0].axis('equal')
ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('z')

ax[0,1].scatter(X[:,1],X[:,2],alpha=0.5)
ax[0,1].plot(W[:,1],W[:,2],'-rx')
ax[0,1].plot(W_[:,1],W_[:,2],'-yx')
ax[0,1].axis('equal')
ax[0,1].set_xlabel('y')
ax[0,1].set_ylabel('z')

ax[1,0].scatter(X[:,0],X[:,1],alpha=0.5)
ax[1,0].plot(W[:,0],W[:,1],'-rx')
ax[1,0].plot(W_[:,0],W_[:,1],'-yx')
ax[1,0].axis('equal')
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('y')

[fig, ax]=pyplot.subplots()
fig.suptitle('Estimated r')
ax.plot(t,r)
ax.axis('equal')
ax.set_xlabel('t [linear coord inflated path]')
ax.set_ylabel('r')

[fig, ax]=pyplot.subplots()
fig.suptitle('Profile of Inflated Path')
for i in range(1,NC+2):
    ax.add_artist(pyplot.Circle((t[i],0),r[i]))
ax.plot(t,np.zeros(t.size),'-rx')
ax.axis('equal')

pyplot.show()

raw_input('Press [enter] to quit')
