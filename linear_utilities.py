import numpy as np

from scipy.spatial import distance

from matplotlib import pyplot

import time
    

def initMedoids(X, n, init_type, exclude_ids=[]): 
    """
    Initialize NC medoids with init_type rational.

    Args:
        [ndarray float] X: data matrix

        [int] n: number of medoids to be selected
        
        [string] init_type: rational to be used
            'uniform': randomly selected with uniform distribution
            'kpp': k-means++ algorithm

        [ndarray int] exclude_ids: blacklisted ids that shouldn't be selected

    Returns:
        [ndarray int] med_ids: indices of the medoids selected
    """

    N=X.shape[0]
    D=X.shape[1]
    med_ids=-1*np.ones(n,int)

    if(init_type=='uniform'):
        while(n>0):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(med_ids==med_id)==0 and np.count_nonzero(exclude_ids==med_id)==0):
                med_ids[n-1]=med_id
                n = n-1

    elif(init_type=='kpp'):
        accepted = False
        while(not accepted):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(exclude_ids==med_id)==0):
                accepted = True
        med_ids[0]=med_id

        for i in range(1,n):
            Xmed_dst = distance.cdist(X,np.vstack([X[med_ids[0:i],:],X[exclude_ids,:]]),'sqeuclidean') 
            D2 = Xmed_dst.min(1)
            D2_n = 1.0/np.sum(D2)
            accepted = False
            while(not accepted):
                med_id = np.random.randint(0,N)
                if(np.random.rand()<D2[med_id]*D2_n):
                    accepted = True
            med_ids[i]=med_id
    else:
        raise ValueError('init_type not recognized.')

    return(med_ids)


def getMouseSamples2D(X, n):
    """
    Get n points from X by manual mouse selection on the first 2 dimensions.

    Args:
        [ndarray float] X: data matrix 

        [int] n: number of points to be selected

    Returns:
        [ndarray int]  ids: indices of the samples selected
    """
    ids=np.ndarray(n,int)
    n_sel=[n]

    [fig,ax]=pyplot.subplots()
    ax.plot(X[:,0],X[:,1],'bo')
    ax.axis('equal')
    pyplot.title('Select '+repr(n)+' points')

    def onclick(ev):
        if(n_sel[0]==0):
            pyplot.close()
            return

        n_sel[0]=n_sel[0]-1
        
        mouseX=np.asarray([ev.xdata,ev.ydata],float)
        id_sel=np.argmin(np.diag(np.matmul(X[:,0:2],X[:,0:2].T))-2*np.matmul(mouseX,X[:,0:2].T))

        ax.plot(X[id_sel,0],X[id_sel,1],'ro')
        if(n_sel[0]==0):
            pyplot.title('Click to quit or close the figure')

        pyplot.draw()

        ids[n_sel[0]]=id_sel;

    cid=fig.canvas.mpl_connect('button_press_event',onclick)
    pyplot.show(fig)
    fig.canvas.mpl_disconnect(cid)

    return ids


def find_elbow(f):
    """
    Find the elbow in a function f, as the point on f with max distance from the line connecting f[0,:] and f[-1,:]

    Args:
        [ndarray float] f: function (Nx2 array in the form [x,f(x)]) 

    Returns:
        [int]  elb_id: index of the elbow 
    """
    ps = np.asarray([f[0,0],f[0,1]])
    pe = np.asarray([f[-1,0],f[-1,1]])
    p_line_dst = np.ndarray(f.shape[0]-2,float)
    for i in range(1,f.shape[0]-1):
        p = np.asarray([f[i,0],f[i,1]])
        p_line_dst[i-1] = np.linalg.norm(np.cross(pe-ps,ps-p))/np.linalg.norm(pe-ps)
    elb_id = np.argmax(p_line_dst)+1

    return elb_id
