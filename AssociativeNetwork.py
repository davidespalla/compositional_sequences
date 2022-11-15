import numpy as np
from dataclasses import dataclass


@dataclass
class Network():
    J: np.array
    N: int = None
    transfer_func: callable = None
    dynamic_func: callable = None
    history: np.array = None

    def __post_init__(self):
        self.N = self.J.shape[0]

    def run_dynamics(self, start_config: np.array,
                     n_steps: int,
                     sparsity: float):
        '''
        Runs a dynamics starting for the given configuration, for the given number of steps,
        constraining the activity to the given sparsity.
        '''
        self.history = np.empty((n_steps, self.N))
        self.history[0, :] = start_config
        for i in range(1, n_steps):
            self.history[i, :] = self.dynamic_func(
                self.history[i-1], self.J, sparsity)

# FUNCTIONS


def net_dynamics(V, J, f):
    '''
    Defines how an iteration step of the dynamics is calculated.    
    '''
    h = np.dot(J, V)
    v_out = np.asarray(list(map(lambda h: ReLu(h), h)))

    # fixes fracion of active neurons
    v_out = fix_sparsity(v_out, f)

    # fixes mean to 1
    v_out = v_out/np.mean(v_out)

    return v_out


def ReLu(x):
    '''
    Returns x if x>0, else returns 0.  
    '''
    if x > 0:
        out = x
    else:
        out = 0

    return out


def fix_sparsity(V, f):
    '''
    Fixes the sparsity of the given vector `V`  by computing the `f` percentile
    of the values of `V`, subtracting it from all values of `V` and setting to 0 
    all elements <0.
    '''
    vout = V
    th = np.percentile(V, (1.0-f)*100)
    for i in range(len(V)):
        if vout[i] < th:
            vout[i] = 0
        else:
            vout[i] = vout[i]-th
    return vout


def calculate_coherence(activity, J_pattern):
    '''
    Computes the coherence of the given activity vector with the given pattern interaction matrix.
    '''

    overlap = activity @ J_pattern @ activity.T
    norm = float(len(activity)*(len(activity)-1)/2.)
    return overlap/norm
