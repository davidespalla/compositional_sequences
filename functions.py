import numpy as np
import PatternStructure as ps
import AssociativeNetwork as nt
import matplotlib.pyplot as plt


def calculate_coherence(activity, J_pattern):
    overlap = activity @ J_pattern @ activity.T
    norm = float(len(activity)*(len(activity)-1)/2.)
    return overlap/norm


def coherence_timecourse(pattern_struct, history):
    coherence = np.zeros((len(pattern_struct.patterns), len(history)))
    for i in range(len(coherence)):
        for t in range(coherence.shape[1]):
            coherence[i, t] = calculate_coherence(
                history[t], pattern_struct.pattern_matrices_symmetric[i])

    return coherence


def attractors_timecourse(pattern_struct, history):
    coherence = coherence_timecourse(pattern_struct, history)
    attractor_tc = np.argmax(coherence, axis=0)
    return attractor_tc


def retrieved_sequence(pattern_struct, history):
    attractor_tc = attractors_timecourse(pattern_struct, history)
    sequence = []
    sequence.append(attractor_tc[0])
    for i in range(1, len(attractor_tc)):
        if attractor_tc[i] != attractor_tc[i-1]:
            sequence.append(attractor_tc[i])
    return np.asarray(sequence)


def compute_retrieval_quality(coherence):
    return np.mean(np.max(coherence, axis=0))


def compute_retrieval_probability(coherence, threshold):
    '''
    Computes the fraction of times during a dynamics in which the maximum coherence was above a given threshold
    '''
    max_coherence = np.max(coherence, axis=0)
    return np.sum(max_coherence > threshold) / len(max_coherence)


def compute_obedience(sequence, transition_mat):
    if len(sequence) <= 1:
        return np.nan

    followed = 0
    for i in range(1, len(sequence)):
        if transition_mat[sequence[i-1], sequence[i]] > 0:
            followed += 1
    return followed / (len(sequence)-1)
