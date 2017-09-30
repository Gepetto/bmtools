from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqs
import numpy as np
import pinocchio as se3
from IPython import embed


def butter_lowpass(cutOff, fs, order=4):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filtfilt_butter(data, cutOff, fs, order=4):
    data= np.array(data).squeeze()
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def centralDiff1(x,y):
    '''
    dx/dy
    x : numpy 1D array
    y : numpy 1D array
    '''
    nx = x.size 
    #interior
    for i in range(1,nx-1):
        slices[i] = np.float64(x[i+1]-x[i-1])
        dy[i] = (y[i+1]-y[i-1]) / slices[i] 
    #boundaries
    dy[0] =  ( y[1] - y[0] ) / ( x[1] - x[0] )
    dy[-1] = ( y[-1] - y[-2] ) / ( x[-1] - x[-2] )

def centralDiff2(y,h):
    '''
    derivative of y with h spacing
    x : numpy 1D array
    h : spacing constant
    '''
    nx = y.size 
    #interior
    for i in range(1,nx-1):
        dy[i] = (y[i+1]-y[i-1]) / (2*h)
    #boundaries
    dy[0] =  ( y[1] - y[0] ) / h
    dy[-1] = ( y[-1] - y[-2] ) / h

def findJointVIdx(model,idx_q):
    jv = []
    for s in range(0, 26):
        idx_v = model.joints[s].idx_v
        nv = model.joints[s].nv
        jv += [[s, idx_v, nv]]
    
    for s in range(0, 26):
        if idx_q >= jv[s][1] and idx_q < (jv[s][1] + jv[s][2]):
            return s

def getCanonicalBaseV(jmodel):
    v = np.zeros(robot.nv)
    if jmodel.shortname() == 'JointModelRevoluteUnaligned':
        v[jmodel.idx_v] = 1
    elif jmodel.shortname() == 'JointModelFreeFlyer':
        v[jmodel.idx_v:jmodel.idx_v+jmodel.nv] = np.ones(jmodel.nv)
    elif jmodel.shortname() == 'JointModelSpherical':
        v[jmodel.idx_v:jmodel.idx_v+jmodel.nv] = np.ones(jmodel.nv)
    elif jmodel.shortname() == 'JointModelRX':
        v[jmodel.idx_v] = 1
    elif jmodel.shortname() == 'JointModelRY':
        v[jmodel.idx_v] = 1
    else:
        print 'no joint model'
    return v

def get_ddJ_ddq(model, data, q, v, h, local):
    '''
    get the tensor d(J_dot)/d(q_dot)
    '''
    q0 = q.copy()
    v0 = v.copy()
    se3.computeJacobiansTimeVariation(model, data, q, v)
    #J_ref = se3.getJacobianTimeVariation(model, data, j, local).copy()
    ddJddq = np.zeros((6, model.nv, model.nv))
    for j in range(1,model.njoints):
        se3.computeJacobiansTimeVariation(model, data, q0, v0)
        J_ref = se3.getJacobianTimeVariation(model, data, j, local).copy()
        vh = np.ones(model.nv)*h
        se3.computeJacobiansTimeVariation(model, data, q, vh)
        J = se3.getJacobianTimeVariation(model, data, j, local).copy()
        ddJddq[:,:,j] = (J - J_ref)/h
    return ddJddq



def get_dJi_dq(model, data, q, joint_id, h, local):
    '''
    get the tensor: dJi/dq 
    '''
    q0 = q.copy()
    se3.forwardKinematics(model, data, q0)
    dJidq = finiteDifferencesdJi_dq(model, data, q0, h, joint_id, local)
    return dJidq


def finiteDifferencesdJi_dq(model, data, q, h, joint_id, local):
    #dJi/dq
    q0 = q.copy()
    se3.forwardKinematics(model, data, q0)
    se3.computeJacobians(model, data, q0)
    tensor_dJi_dq = np.zeros((6, model.nv,model.nv))
    J0i = se3.jacobian(model, data, q0, joint_id, local, True).copy()
    oMi = data.oMi[joint_id].copy()
    for j in range(model.nv):
        vh = np.matrix(np.zeros((model.nv,1))); 
        vh[j] = h #eps
        q_integrate = se3.integrate(model, q0.copy(), vh) # dJk/dqi = q0 + v*h
        se3.forwardKinematics(model, data, q_integrate)
        se3.computeJacobians(model, data, q_integrate)
        oMint = data.oMi[joint_id].copy()
        iMint = oMi.inverse() * oMint
        J0_int = se3.jacobian(model, data, q_integrate, joint_id, local, True).copy()
        J0_int_i = J0_int.copy()
        J0_int_i = iMint.action*J0_int_i
        tensor_dJi_dq[:,:,j] = (J0_int_i-J0i) / h
    return tensor_dJi_dq

def get_dA_dq(model, data, q, v, h):
    '''
    get the tensor: dA/dq 
    '''
    q0 = q.copy()
    v0 = v.copy() 
    dAidq = finiteDifferencesdA_dq(model, data, q0, v0, h)
    return dAidq


def finiteDifferencesdA_dq(model, data, q, v, h):
    '''
    dAi/dq
    '''
    q0 = q.copy()
    v0 = v.copy()
    tensor_dAi_dq = np.zeros((6, model.nv,model.nv))
    se3.forwardKinematics(model, data, q0, v0)
    se3.computeJacobians(model, data, q0)
    pcom_ref = se3.centerOfMass(model, data, q0).copy()
    se3.ccrba(model, data, q0, v0)
    A0i = data.Ag.copy()
    oMc_ref = se3.SE3.Identity()#data.oMi[1].copy() 
    oMc_ref.translation = pcom_ref#oMc_ref.translation - pcom_ref
    for j in range(model.nv):
        #vary q
        vh = np.matrix(np.zeros((model.nv,1)));  vh[j] = h 
        q_integrated = se3.integrate(model, q0.copy(), vh) 
        se3.forwardKinematics(model, data, q_integrated)
        se3.computeJacobians(model, data, q_integrated)
        pcom_int = se3.centerOfMass(model, data, q_integrated).copy()
        se3.ccrba(model, data, q_integrated, v0)
        A0_int = data.Ag.copy()
        oMc_int = se3.SE3.Identity() #data.oMi[1].copy()
        oMc_int.translation = pcom_int #oMc_int.translation - pcom_int
        cMc_int = oMc_ref.inverse() * oMc_int
        A0_int = cMc_int.dualAction * A0_int
        tensor_dAi_dq[:,:,j] = (A0_int-A0i) / h
    return tensor_dAi_dq













def get_ddA_dq(model, data, q, v, h):
    '''
    get the tensor: ddA/dq 
    '''
    q0 = q.copy()
    v0 = v.copy() 
    ddAdq = finiteDifferencesddA_dq(model, data, q0, v0, h)
    return ddAdq

def finiteDifferencesddA_dq(model, data, q, v, h):
    '''
    d(A_dot)/dq
    '''
    q0 = q.copy()
    v0 = v.copy()
    tensor_ddA_dq = np.zeros((6, model.nv,model.nv))
    se3.forwardKinematics(model, data, q0, v0)
    se3.computeJacobians(model, data, q0)
    #se3.centerOfMass(model, data, q0)
    pcom_ref = se3.centerOfMass(model, data, q0).copy()
    vcom_ref = data.vcom[0].copy()
    #se3.ccrba(model, data, q0, v0.copy())
    se3.dccrba(model, data, q0, v0.copy())
    dA0 =  np.nan_to_num(data.dAg).copy()
    oMc_ref = se3.SE3.Identity() 
    #oMc_ref.translation = pcom_ref  
    oMc_ref.translation = vcom_ref  
    for j in range(model.nv):
        #vary q
        vh = np.matrix(np.zeros((model.nv,1)));  vh[j] = h 
        q_integrated = se3.integrate(model, q0.copy(), vh) 
        se3.forwardKinematics(model, data, q_integrated)#, v0.copy())
        #se3.computeJacobians(model, data, q_integrated)
        #se3.centerOfMass(model, data, q_integrated)
        pcom_int = se3.centerOfMass(model, data, q_integrated).copy()
        vcom_int = data.vcom[0].copy()
        #se3.ccrba(model, data, q_integrated, v0.copy())
        se3.dccrba(model, data, q_integrated, v0.copy())
        dA0_int = np.nan_to_num(data.dAg).copy()
        oMc_int = se3.SE3.Identity() 
        #oMc_int.translation = pcom_int 
        oMc_int.translation = vcom_int 
        cMc_int = oMc_ref.inverse() * oMc_int
        dA0_int = cMc_int.dualAction * dA0_int
        tensor_ddA_dq[:,:,j] = (dA0_int-dA0) / h
    return tensor_ddA_dq






def get_ddA_ddq(model, data, q, v, h):
    '''
    get the tensor: ddA/ddq 
    '''
    q0 = q.copy()
    v0 = v.copy() 
    ddAddq = finiteDifferencesddA_dq(model, data, q0, v0, h)
    return ddAddq

def finiteDifferencesddA_ddq(model, data, q, v, h):
    '''
    d(A_dot)/d(q_dot)
    '''
    q0 = q.copy()
    v0 = v.copy()
    tensor_ddA_ddq = np.zeros((6, model.nv,model.nv))
    se3.forwardKinematics(model, data, q0, v0)
    vcom_ref = data.vcom[0].copy()
    se3.dccrba(model, data, q0, v0)
    dA0 = np.nan_to_num(data.dAg).copy()
    oMc_ref = se3.SE3.Identity()
    oMc_ref.translation = vcom_ref
    for j in range(model.nv):
        # vary dq
        ah = np.matrix(np.zeros((model.nv,1)));  ah[j] = h 
        v_new = v0 + ah
        se3.forwardKinematics(model, data, q0, v_new)
        vcom_new = data.vcom[0].copy()
        se3.dcrba(model, data, q0, v_new)
        dA0i_int = np.nan_to_num(data.dAg).copy()
        oMc_int = se3.SE3.Identity() 
        oMc_int.translation = vcom_new 
        cMc_int = oMc_ref.inverse() * oMc_int
        dA0_int = cMc_int.dualAction * dA0_int
        tensor_ddA_ddq[:,:,j] = (dA0_int-dA0) / h
    return tensor_dA_dq


def prodVecTensor(vec, tensor):
    '''
    vec.T * Matrix
    '''
    nk = tensor.shape[0]
    nv = tensor.shape[2]
    #assert nv == tensor.shape[2]
    D  = np.zeros((nk,nv))
    for i in xrange(nk):
        for j in xrange(nv):
            for k in xrange(nv):
                D[i][j] +=  tensor[i,j,k] * vec[k]                                                           
    return D


'''
def computePartialDerivatives(obot, q, v, J, dJ):
    # Does not work ,it can bbe used for matrix 
    # dim(J) = i x k
    # dim(dq) = j = k 
    nv = robot.nv # k
    D  = np.zeros((6,nv))
    for i in xrange(6):
        for j in xrange(nv):
            for k in xrange(nv):
                D[i][j] +=  T[i,k,j] * dq[k]
    return D
'''    

def computeFirstSecondDerivatives(model, q, time):
    """
    Return the first and second order derivatives of q.

    The first oreder numerical derivate of q is computed using
    forward differences for q[0], central differences for
    q[1:-2] and barckard differences for q[-1].
    The second order numerical derivative is computed using
    numpy.gradient which uses second order central differences
    in the interior and forward/backward differences at the
    boundaries (just like the firs order derivator).

    Parameters
    ----------
    model: pinocchio.model
      A pinocchio model 
    q: numpy matrix q[time, Ncoordinates] 
      q is element of the configuration space and represents 
      the generalized coordinates.
    time: numpy matrix     
      time is a column matrix containing the time slices

    Returns
    -------
    dq: numpy matrix dq[time, model.nv]
      dq is a matrix that represents the tangent space of q
    ddq: numpy matrix ddq[time, model.nv]
      ddq is a matrix that represents the tangent space of dq
    
    """
    t = np.asarray(time).squeeze()
    tmax, ncoord = q.shape
    
    # Numerical differentiation: 1st order                        
    dq = np.asmatrix(np.zeros([tmax, model.nv]))
    tslices = np.zeros(tmax)
    tslices[0] = np.float64(t[1]-t[0])
    tslices[-1] = np.float64(t[-1]-t[-2])

    #for i in range(1,tmax-1):
    #    tslices[i] = np.float64(t[i+1]-t[i-1])
    #    dq[i] = (se3.differentiate(model, q[i-1],  q[i+1]) / tslices[i]).T
        
    
    # interior
    for i in range(1,tmax-1):
        tslices[i] = np.float64(t[i+1]-t[i-1])
        dq[i] = (se3.differentiate(model, q[i-1],  q[i+1]) / tslices[i]).T
    
    #TODO -- slices are 2dt -- change to 1dt
    # boundaries 
    dq[0] = (se3.differentiate(model, q[0],  q[1]) / tslices[0]).T
    dq[-1] = (se3.differentiate(model, q[-2],  q[-1]) / tslices[-1]).T
    
    # Numerical differentiation: 2nd order
    ddq = np.asmatrix(np.zeros([tmax, model.nv]))
    for q in range(0, model.nv):
        data = np.asmatrix(np.gradient(dq[:,q].A1, tslices)).T
        y=filtfilt_butter(data, 35, 400, 4)
        ddq[:,q] = np.matrix(y).T
        #ddq[:,q] = np.asmatrix(np.gradient(dq[:,q].A1, tslices)).T
    
    return dq, ddq


def diffM(M, time):
    '''
    Numerical differentiation of a matrix
    M : list containing a numpy matrix 
    '''
    M = np.asarray(M)
    
    (tmax,n,m) = M.shape #100x6x42
    t = np.asarray(time).squeeze()
    #tmax = len(Jtask)
    tslices = np.zeros(tmax)
    tslices[0] = np.float64(t[1]-t[0])
    tslices[-1] = np.float64(t[-1]-t[-2])

    for f in range(1,tmax-1):
        tslices[f] = np.float64(t[f+1]-t[f-1])
        
    dM = np.asarray(np.zeros([tmax,n,m]))
    dMlist = []
    for i in xrange(n):
        for j in xrange(m):
            dM[:,i,j] = np.matrix(np.gradient(M[:,i,j], tslices)).T.squeeze()
    dMlist.append(dM)

    [item for sublist in dMlist for item in sublist]
    return sublist
    
# Basic Statistics

def statsQSE3(model, q):
    # computes means, stds
    pass

def statsArray(A):
    pass

def statsMatrix(M):
    pass
