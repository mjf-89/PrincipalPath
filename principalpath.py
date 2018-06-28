import numpy as np

from scipy.spatial import distance
from scipy.sparse import csgraph

from matplotlib import pyplot
from matplotlib.widgets import Slider, Button, RadioButtons

import linear_utilities as lu


def rkm(X, init_W, s, plot_ax=None):
    """
    Regularized K-means for principal path, MINIMIZER.

    Args:
        [ndarray float] X: data matrix

        [ndarray float] init_W: initial waypoints matrix

        [float] s: regularization parameter 

        [matplotlib.axis.Axes] plot_ax: Axes for the 2D plot (first 2 dim of X), None to avoid plotting

    Returns:
        [ndarray float] W: final waypoints matrix

        [ndarray int] labels: final

    References:
        [1] 'Finding Prinicpal Paths in Data Space', M.J.Ferrarotti, W.Rocchia, S.Decherchi, [submitted]
        [2] 'Design and HPC Implementation of Unsupervised Kernel Methods in the Context of Molecular Dynamics', M.J.Ferrarotti, PhD Thesis.
    """

    #extract useful info from args
    N = X.shape[0]
    d = X.shape[1]
    NC = init_W.shape[0]-2

    #construct boundary matrix
    boundary = init_W[[0,NC+1],:]
    B=np.zeros([NC,d],float)
    B[[0,NC-1],:]=boundary

    #construct regularizer hessian
    AW = np.diag(np.ones(NC))+np.diag(-0.5*np.ones(NC-1),1)+np.diag(-0.5*np.ones(NC-1),-1)

    #compute initial labels
    XW_dst = distance.cdist(X,init_W,'sqeuclidean')
    u = XW_dst.argmin(1)

    #iterate the minimizer
    converged = False
    it = 0
    while(not converged):
        it = it+1
        print('iteration '+repr(it))

        #compute cardinality
        W_card=np.zeros(NC+2,int)
        for i in range(NC+2):
            W_card[i] = np.sum(u==i)

        #compute centroid matrix
        C = np.ndarray([NC,d],float)
        for i in range(NC):
            C[i,:] = np.sum(X[u==i+1,:],0)

        #construct k-means hessian 
        AX = np.diag(W_card[1:NC+1])

        #update waypoints
        W = np.matmul(np.linalg.pinv(AX+s*AW),C+0.5*s*B)
        W = np.vstack([boundary[0,:],W,boundary[1,:]])

        #compute new labels
        XW_dst = distance.cdist(X,W,'sqeuclidean')
        u_new = XW_dst.argmin(1)

        #check for convergence
        converged = not np.sum(u_new!=u)
        u=u_new

        #plot
        if(plot_ax is not None):
            pyplot.sca(plot_ax)
            pyplot.ion()
            pyplot.cla()
            pyplot.title('Annealing, s='+repr(s))
            pyplot.plot(X[:,0],X[:,1],'bo')
            pyplot.plot(W[:,0],W[:,1],'-ro')
            pyplot.axis('equal')

            pyplot.pause(1.0/60)
    
    return W, u


def rkm_cost(X, W, s):
    """
    Regularized K-means for principal path, COST EVALUATION.
    (most stupid implementation)

    Args:
        [ndarray float] X: data matrix

        [ndarray float] W: waypoints matrix

        [float] s: regularization parameter 

    Returns:
        [float] cost_km: K-means part of the cost

        [float] cost_reg: regularizer part of the cost
    """

    XW_dst = distance.cdist(X,W,'sqeuclidean')
    u = XW_dst.argmin(1)

    cost_km=0.0
    for i,x in enumerate(X):
        w = W[u[i],:]
        cost_km = cost_km + np.dot(x,x) + np.dot(w,w) -2*np.dot(x,w)

    cost_reg=0.0
    for i,w in enumerate(W[0:-1,:]):
        w_nxt = W[i+1,:]
        cost_reg = cost_reg + np.dot(w,w) + np.dot(w_nxt,w_nxt) - 2*np.dot(w,w_nxt)
    cost_reg = s*cost_reg

    return cost_km, cost_reg
    


def rkm_prefilter(X, boundary_ids, Nf=200, k=5, p=1000, T=0.1, plot_ax=None):
    """
    Regularized K-means for principal path, PREFILTER.

    Args:
        [ndarray float] X: data matrix

        [ndarray int] boundary_ids: start/end waypoints as sample indices

        [int] Nf: number of filter centroids

        [int] k: number of nearest neighbor for the penalized graph

        [float] p: penalty factor for the penalized graph

        [float] T: filter threshold

        [matplotlib.axis.Axes] plot_ax: Axes for the 2D plot (first 2 dim of X), None to avoid plotting

    Returns:
        [ndarray float] X_filtered

        [ndarray int] boundary_ids_filtered

        [ndarray float] X_garbage
    """

    #pick Nf medoids with k-means++ and compute pairwise distance matrix
    med_ids = lu.initMedoids(X, Nf-2, 'kpp', boundary_ids)
    med_ids = np.hstack([boundary_ids[0],med_ids,boundary_ids[1]])
    medmed_dst = distance.cdist(X[med_ids,:],X[med_ids,:],'sqeuclidean')

    #build k-nearest-neighbor penalized matrix
    knn_ids = np.argsort(medmed_dst,1)
    medmed_dst_p = medmed_dst.copy()*p
    for i in range(Nf):
        for j in range(k):
            k=knn_ids[i,j]
            medmed_dst_p[i,k] = medmed_dst[i,k]
            medmed_dst_p[k,i] = medmed_dst[k,i]
    medmed_dst_p[0,Nf-1]=0
    medmed_dst_p[Nf-1,0]=0

    #find shortest path using dijkstra
    [path_dst, path_pre] = csgraph.dijkstra(medmed_dst_p, False, 0,True)
    path=np.ndarray(0,int)
    i=Nf-1
    while(i != 0):
        path=np.hstack([i,path])
        i = path_pre[i]
    path=np.hstack([i,path])

    #filter out medoids too close to the shortest path
    T=T*np.mean(medmed_dst)

    to_filter_ids=np.ndarray(0,int)
    for i in path:
        to_filter_ids = np.hstack([np.where(medmed_dst[i,:]<T)[0], to_filter_ids])
    to_filter_ids = np.setdiff1d(to_filter_ids,path)
    to_filter_ids = np.unique(to_filter_ids)

    to_keep_ids = np.setdiff1d(np.asarray(range(Nf)),to_filter_ids)

    Xmed_dst = distance.cdist(X,X[med_ids[to_keep_ids],:],'sqeuclidean')
    u = med_ids[to_keep_ids][Xmed_dst.argmin(1)]

    N=X.shape[0]
    filter_mask = np.zeros(N,bool)
    for i in range(N):
        if u[i] in med_ids[path]:
            filter_mask[i]=True
    
    #convert boundary indices
    boundary_ids_filtered = boundary_ids.copy()
    boundary_ids_filtered[0] = boundary_ids[0] - boundary_ids[0] + np.sum(filter_mask[0:boundary_ids[0]])
    boundary_ids_filtered[1] = boundary_ids[1] - boundary_ids[1] + np.sum(filter_mask[0:boundary_ids[1]])

    #plot filter figure 
    if(plot_ax is not None):
        pyplot.sca(plot_ax)
        pyplot.ion()
        pyplot.plot(X[np.logical_not(filter_mask),0],X[np.logical_not(filter_mask),1],'yo',label='data filtered out')
        pyplot.plot(X[filter_mask,0],X[filter_mask,1],'bo',label='data kept')
        pyplot.plot(X[med_ids,0],X[med_ids,1],'ro',label='filter medoids')
        pyplot.plot(X[med_ids[to_filter_ids],0],X[med_ids[to_filter_ids],1],'kx',label='filter medoids dropped')
        pyplot.plot(X[med_ids[path],0],X[med_ids[path],1],'-go',label='filter shortest path')
        pyplot.plot(X[filter_mask,:][boundary_ids_filtered,0],X[filter_mask,:][boundary_ids_filtered,1],'mo',label='boundary samples')
        pyplot.legend()
        pyplot.axis('equal')

    return X[filter_mask,:], boundary_ids_filtered, X[np.logical_not(filter_mask),:]


def rkm_MS_evidence(models, s_span, X):
    """
    Regularized K-means for principal path, MODEL SELECTION, Bayesian Evidence.

    Args:
        [ndarray float] models: matrix with path models, shape N_models x N x (NC+2)

        [ndarray float] s_span: array with values of the reg parameter for each model (sorted in decreasing order, with 0 as last value)

        [ndarray float] X: data matrix

    Returns:
        [ndarray float] logE_s: array with values of log evidence for each model
    """

    if(s_span[-1]>0.0):
        raise ValueError('In order to evaluate the evidence a model with s=0 has to be provided')

    #Evaluate unregularized cost
    cost_ureg=np.sum(rkm_cost(X, models[-1,:,:],s_span[-1]))

    logE_s = np.ndarray(s_span.size,float)
    for i,s in enumerate(s_span):
        N = X.shape[0]
        W = models[i,:,:]
        NC = W.shape[0]-2
        d = W.shape[1]

        #Set gamma (empirical rational) and compute lambda
        gamma = np.sqrt(N)*0.125/np.mean(distance.cdist(X,X,'euclidean'))
        lambd = s*gamma 

        #Maximum Posterior cost
        cost_MP=np.sum(rkm_cost(X, W, s))

        #Find labels
        XW_dst = distance.cdist(X,W,'sqeuclidean')
        u = XW_dst.argmin(1)
        #Compute cardinality
        W_card=np.zeros(NC+2,int)
        for j in range(NC+2):
            W_card[j] = np.sum(u==j)

        #Construct boundary matrix
        boundary = W[[0,NC+1],:]
        B=np.zeros([NC,d],float)
        B[[0,NC-1],:]=boundary

        #Construct regularizer hessian
        AW = np.diag(np.ones(NC))+np.diag(-0.5*np.ones(NC-1),1)+np.diag(-0.5*np.ones(NC-1),-1)

        #Construct k-means hessian 
        AX = np.diag(W_card[1:NC+1])

        #Compute global hessian
        A = AX+s*AW

        #Evaluate log-evidence
        logE = -0.5*d*np.log(np.sum(np.linalg.eigvals(A)))
        logE = logE + gamma*(cost_ureg-cost_MP)
        if(lambd>0):
            logE = logE + 0.5*d*NC*np.log(lambd)
        else:
            logE = logE + 0.5*d*NC*np.log(lambd+np.finfo(np.float).eps)

        logE = logE - 0.125*lambd*np.trace(np.matmul(B.T,np.matmul(np.linalg.pinv(AW),B)))
        logE = logE + 0.25*lambd*np.trace(np.matmul(B.T,B))

        logE_s[i] = logE

    return logE_s


def rkm_MS_pathlen(models, s_span, X):
    """
    Regularized K-means for principal path, MODEL SELECTION, Path length.

    Args:
        [ndarray float] models: matrix with path models, shape N_models x N x (NC+2)

        [ndarray float] s_span: array with values of the reg parameter for each model (sorted in decreasing order, with 0 as last value)

        [ndarray float] X: data matrix

    Returns:
        [ndarray float] len_s: array with values of path length for each model
    """
    len_s=np.zeros(s_span.size,float)
    for i,s in enumerate(s_span):
        W = models[i,:,:]
        NC = W.shape[0]-2
        for j,w in enumerate(W[0:-1,:]):
            w_nxt = W[j+1,:]
            len_s[i] = len_s[i] + np.sqrt(np.dot(w,w)+np.dot(w_nxt,w_nxt)-2*np.dot(w,w_nxt))

    return len_s


def rkm_MS_pathvar(models, s_span, X):
    """
    Regularized K-means for principal path, MODEL SELECTION, variance on waypoints interdistance.

    Args:
        [ndarray float] models: matrix with path models, shape N_models x N x (NC+2)

        [ndarray float] s_span: array with values of the reg parameter for each model (sorted in decreasing order, with 0 as last value)

        [ndarray float] X: data matrix

    Returns:
        [ndarray float] W_dst_var: array with values of variance for each model
    """
    W_dst_var=np.ndarray(models.shape[0],float)
    for i in range(models.shape[0]):
        W = models[i,:,:]
        W_dst=np.linalg.norm(W[1:,:]-W[0:-1,:],axis=1)
        W_dst_var[i] = np.var(W_dst)

    return W_dst_var


def rkm_MS_ksgm(models, s_span, X):
    """
    Regularized K-means for principal path, MODEL SELECTION, k-segment projection error.

    Args:
        [ndarray float] models: matrix with path models, shape N_models x N x (NC+2)

        [ndarray float] s_span: array with values of the reg parameter for each model (sorted in decreasing order, with 0 as last value)

        [ndarray float] X: data matrix

    Returns:
        [ndarray float] ksgm_s: array with values of k-segment projection error for each model
    """
    N = X.shape[0]
    KX = np.matmul(X,X.T)
    ksgm_s = np.zeros(models.shape[0],float)
    for i in range(models.shape[0]):
        W = models[i,:,:]
        NC = W.shape[0]

        KW = np.matmul(W,W.T)
        KXW = np.matmul(X,W.T)

        a2 = np.tile(np.diag(KX)[:,np.newaxis],[1,NC-1]) + np.tile(np.diag(KW)[:-1],[N,1]) - 2*KXW[:,:-1]
        b2 = np.diag(KW)[:-1]+np.diag(KW)[1:]-2*np.diag(KW,1)
        ab = KXW[:,1:]-KXW[:,:-1]+np.tile(np.diag(KW)[:-1],[N,1])-np.tile(np.diag(KW,1),[N,1])
        if(np.all(b2>0)):
            dst2 = a2 - ab*ab / b2
        else:
            dst2 = a2 - ab*ab / (b2+np.finfo(np.float).eps)

        prj_mask = np.logical_and(ab>0,ab<b2)
        dst2[prj_mask==0] = np.inf
        prj_mask = np.max(prj_mask,1)

        dst2_line = np.min(dst2,1)
        dst2_vrtx = np.min(distance.cdist(X,W,'sqeuclidean'),1)

        ksgm_s[i] = np.sum(dst2_line[prj_mask])+np.sum(dst2_vrtx[prj_mask==0])

    return ksgm_s


def rkm_MS_gui(models, s_span, X, X_g=None):
    N = X.shape[0]
    d = X.shape[1]
    ####
    #GUI
    ####
        #Main axis (for data)
    pyplot.ion()
    [gui,ax_data] = pyplot.subplots()
    ax_data.set_title('Interactive Model Exploration')
    pyplot.subplots_adjust(0.25,0.25,0.75,0.9)

        #buttons to perform MS
    ax_MS_ev_btn = pyplot.axes([0.8, 0.85, 0.2, 0.05])
    MS_ev_btn = Button(ax_MS_ev_btn, 'MS: evidence')

    ax_MS_ksgm_btn = pyplot.axes([0.8, 0.75, 0.2, 0.05])
    MS_ksgm_btn = Button(ax_MS_ksgm_btn, 'MS: k-segment')

    ax_MS_len_btn = pyplot.axes([0.8, 0.65, 0.2, 0.05])
    MS_len_btn = Button(ax_MS_len_btn, 'MS: path len')

    ax_MS_var_btn = pyplot.axes([0.8, 0.55, 0.2, 0.05])
    MS_var_btn = Button(ax_MS_var_btn, 'MS: path var')

        #slider to select s
    ax_s_sld = pyplot.axes([0.25, 0.1, 0.5, 0.03])
    ax_s_sld.set_title('[drag to change the value of s]')
    s_sld = Slider(ax_s_sld, 's', 0, s_span.size-1, valstep=1.0)

    ####
    #initial plot
    ####
    [X_plt, ] = ax_data.plot(X[:,0],X[:,1],'bo')
    if(X_g is not None):
        [X_g_plt, ] = ax_data.plot(X_g[:,0],X_g[:,1],'yo')

    s_id=0
    [W_plt,] = ax_data.plot(models[s_id,:,0],models[s_id,:,1],'-ro')
    ax_data.axis('equal')


    ####
    #event handlers
    ####
        #s slider handler
    def s_sld_onchanged(val):
        s_id = int(s_span.size-1-val)

        W_plt.set_data(models[s_id,:,0:2].T)
        s_sld.valtext.set_text("s={:.2f}\ns_id={:d}".format(s_span[s_id],s_id))

        #max evidence button handler
    def MS_ev_btn_onclicked(ev):
        logE_s = rkm_MS_evidence(models, s_span, X)
        s_maxE_id = np.argmax(logE_s)
        s_sld.set_val(s_span.size-1-s_maxE_id)

        [fig,(ax1,ax2)]=pyplot.subplots(2,1)

            #plot evidence vs s
        ax1.set_title('Model Selection with max Evidence')
        ax1.set_xlabel('s')
        ax1.set_ylabel('log(E)')
        ax1.semilogx(np.flip(s_span,0), np.flip(logE_s,0))
        ax1.plot(s_span[s_maxE_id],logE_s[s_maxE_id],'ro')

            #plot model selected
        ax2.plot(X[:,0],X[:,1],'bo')
        if(X_g is not None):
            ax2.plot(X_g[:,0],X_g[:,1],'yo')
        ax2.plot(models[s_maxE_id,:,0],models[s_maxE_id,:,1],'-ro')
        ax2.axis('equal')

        # k-segment projection error button handler
    def MS_ksgm_btn_onclicked(ev):
        ksgm_s = rkm_MS_ksgm(models, s_span, X)
        i=0
        while(i<ksgm_s.size-1 and ksgm_s[i]>ksgm_s[i+1]):
            i=i+1
        s_minksgm_id = i
        s_sld.set_val(s_span.size-1-s_minksgm_id)

            #plot k-segment projection error vs s
        [fig,(ax1,ax2)]=pyplot.subplots(2,1)
        ax1.set_title('Model Selection with min k-segment projection error')
        ax1.set_xlabel('s')
        ax1.set_ylabel('ksgm')
        ax1.semilogx(np.flip(s_span,0), np.flip(ksgm_s,0))
        ax1.plot(s_span[s_minksgm_id],ksgm_s[s_minksgm_id],'ro')

            #plot model selected
        ax2.plot(X[:,0],X[:,1],'bo')
        if(X_g is not None):
            ax2.plot(X_g[:,0],X_g[:,1],'yo')
        ax2.plot(models[s_minksgm_id,:,0],models[s_minksgm_id,:,1],'-ro')
        ax2.axis('equal')

        #elbow criteria on path length button handler
    def MS_len_btn_onclicked(ev):
        len_s = rkm_MS_pathlen(models, s_span, X)
        s_elb_id = lu.find_elbow(np.stack([s_span,len_s],-1))
        s_sld.set_val(s_span.size-1-s_elb_id)

            #plot path length vs s
        [fig,(ax1,ax2)]=pyplot.subplots(2,1)
        ax1.set_title('Model Selection with elbow method on path length')
        ax1.set_xlabel('s')
        ax1.set_ylabel('path length')
        ax1.plot(np.flip(s_span,0), np.flip(len_s,0))
        ax1.plot(s_span[s_elb_id],len_s[s_elb_id],'ro')

            #plot model selected
        ax2.plot(X[:,0],X[:,1],'bo')
        if(X_g is not None):
            ax2.plot(X_g[:,0],X_g[:,1],'yo')
        ax2.plot(models[s_elb_id,:,0],models[s_elb_id,:,1],'-ro')
        ax2.axis('equal')

        #elbow criteria on waypoints distance variance button handler
    def MS_var_btn_onclicked(ev):
        W_dst_var=rkm_MS_pathvar(models, s_span, X)
        s_elb_id = lu.find_elbow(np.stack([s_span,W_dst_var],-1))
        s_sld.set_val(s_span.size-1-s_elb_id)

            #plot waypoints distance variance vs s
        [fig,(ax1,ax2)]=pyplot.subplots(2,1)
        ax1.set_title('Model Selection with elbow method on waypoins distance variance')
        ax1.set_xlabel('s')
        ax1.set_ylabel('W distance variance')
        ax1.plot(np.flip(s_span,0), np.flip(W_dst_var,0))
        ax1.plot(s_span[s_elb_id],W_dst_var[s_elb_id],'ro')

            #plot model selected
        ax2.plot(X[:,0],X[:,1],'bo')
        if(X_g is not None):
            ax2.plot(X_g[:,0],X_g[:,1],'yo')
        ax2.plot(models[s_elb_id,:,0],models[s_elb_id,:,1],'-ro')
        ax2.axis('equal')


    ####
    #register handlers
    ####
    s_sld.on_changed(s_sld_onchanged)
    MS_ev_btn.on_clicked(MS_ev_btn_onclicked)
    MS_ksgm_btn.on_clicked(MS_ksgm_btn_onclicked)
    MS_len_btn.on_clicked(MS_len_btn_onclicked)
    MS_var_btn.on_clicked(MS_var_btn_onclicked)

    s_sld.set_val(s_span.size/2)

    pyplot.show()

    raw_input('select model with GUI then press [enter] to continue')

    return int(s_span.size-1-s_sld.val)
