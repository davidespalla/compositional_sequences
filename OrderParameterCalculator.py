from dataclasses import dataclass
import numpy as np
from PatternStructure import PatternStructure
from AssociativeNetwork import Network
import functions as fs
from itertools import chain


@dataclass
class OrderParameterCalculator:

    memories: PatternStructure
    network: Network

    coherence_timecourse: np.array = None
    attractors_timecourse: np.array = None
    retrieved_sequence: np.array = None

    def __post_init__(self):
        self.coherence_timecourse = fs.coherence_timecourse(
            self.memories, self.network.history)
        self.attractors_timecourse = fs.attractors_timecourse(
            self.memories, self.network.history)
        self.retrieved_sequence = fs.retrieved_sequence(
            self.memories, self.network.history)
        return

    def compute_all_retrieval_probability(self, threshold=1):
        return fs.compute_retrieval_probability(self.coherence_timecourse, threshold=threshold)

    def compute_on_chain_retrieval_probability(self, threshold=1):
        on_chain_attractors = np.unique(
            list(chain.from_iterable(self.memories.chains)))

        on_chain = np.asarray([
            True if i in on_chain_attractors else False for i in self.attractors_timecourse])

        max_coherence = np.max(self.coherence_timecourse, axis=0)
        retrieved = max_coherence > threshold

        rp = np.sum(np.logical_and(on_chain, retrieved))/len(max_coherence)

        return rp

    def compute_off_chain_retrieval_probability(self, threshold=1):
        on_chain_attractors = np.unique(
            list(chain.from_iterable(self.memories.chains)))

        on_chain = np.asarray([
            True if i in on_chain_attractors else False for i in self.attractors_timecourse])

        max_coherence = np.max(self.coherence_timecourse, axis=0)
        retrieved = max_coherence > threshold

        rp = np.sum(np.logical_and(~on_chain, retrieved))/len(max_coherence)

        return rp
