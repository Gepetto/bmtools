from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqs
import numpy as np
import pinocchio as se3

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
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



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
    
    # interior
    for i in range(1,tmax-1):
        tslices[i] = np.float64(t[i+1]-t[i-1])
        dq[i] = (se3.differentiate(model, q[i-1],  q[i+1]) / tslices[i]).T
    
    # boundaries
    dq[0] = (se3.differentiate(model, q[0],  q[1]) / tslices[0]).T
    dq[-1] = (se3.differentiate(model, q[-2],  q[-1]) / tslices[-1]).T
    
    # Numerical differentiation: 2nd order
    ddq = np.asmatrix(np.zeros([tmax, model.nv]))
    for q in range(0, model.nv):
        ddq[:,q] = np.asmatrix(np.gradient(dq[:,q].A1, tslices)).T
    
    return dq, ddq



# Basic Statistics

def statsQSE3(model, q):
    # computes means, stds
    pass

def statsArray(A):
    pass

def statsMatrix(M):
    pass
