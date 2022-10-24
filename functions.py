import numpy as np
import PatternStructure as ps
import AssociativeNetwork as nt
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import ipywidgets as widgets


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
