import random
from dataclasses import dataclass
import numpy as np


class InteractionKernel:

    def __init__(self, gamma, xi):
        self.gamma = gamma
        self.xi = xi

    def symmetric_kernel(self, x1, x2):
        '''
        Symmetric part of the interaction kernel
        '''
        d = x1-x2
        return np.exp(-d**2/self.xi)

    def antisymmetric_kernel(self, x1, x2):
        '''
        Anti-symmetric part of the interaction kernel
        '''
        d = x1-x2
        return -(d/self.xi)*np.exp(-d**2/self.xi)

    def kernel(self, x1, x2):
        '''
        Total interaction kernel
        '''
        return self.symmetric_kernel(x1, x2)-self.gamma*self.antisymmetric_kernel(x1, x2)


@dataclass
class PatternStructure:

    n_cells: int
    n_patterns: int
    cells_per_pattern: int
    n_chains: int
    patterns_per_chain: int
    kernel: InteractionKernel

    build_structure: bool = True

    patterns: list = None
    chains: list = None
    chain_transitions: np.array = None
    pattern_matrices: list = None
    pattern_matrices_symmetric: list = None
    autoassociative_matrix: np.array = None
    heteroassociative_matrix: np.array = None
    interaction_matrix: np.array = None

    def generate_patterns(self, patterns=None):
        '''
        If patterns are not given, randomly generates the patterns of the memory structure 
        by random permutations of the field centers on the line (0,1).

        If patterns are given assigns the given patterns to the memory structure.
        '''
        if patterns == None:
            self.patterns = []
            field_centers = np.linspace(0, 1, self.cells_per_pattern)

            for p in range(self.n_patterns):
                pattern_cells = random.sample(
                    range(self.n_cells), self.cells_per_pattern)
                pattern = {}
                for i, cell in enumerate(pattern_cells):
                    pattern[cell] = field_centers[i]
                self.patterns.append(pattern)
        else:
            self.patterns = patterns

    def generate_chains(self, chains=None):
        '''
        If chains are not given, randomly generates the pattern chains of the memory structure 
        by randomly drawing from the patterns in the structure.

        If chains are given assigns the given chains to the memory structure.
        '''
        if chains == None:
            self.chains = []
            for i in range(self.n_chains):
                self.chains.append(random.sample(
                    range(self.n_patterns), self.patterns_per_chain))
        else:
            self.chains = chains

        self.chain_transitions = np.zeros((self.n_patterns, self.n_patterns))
        for chain in self.chains:
            for i in range(1, len(chain)):
                self.chain_transitions[chain[i-1], chain[i]] += 1

    def build_interactions(self):
        '''
        Builds the total interaction matrix by summing the autoassociative 
        and heteroassociative components
        '''

        if self.autoassociative_matrix == None:
            #print("Building autoassociative interactions ...")
            self._build_autoassociative_matrix_()

        if self.heteroassociative_matrix == None:
            #print("Building heteroassociative interactions ...")
            self._build_heteroassociative_matrix_()

        self.interaction_matrix = self.autoassociative_matrix+self.heteroassociative_matrix

    def _build_pattern_matrix_(self, pattern):
        '''
        Builds the interaction matrix of the specified pattern.
        The pattern-specific interaction matrices are mostly used in the calculations
        of the order parameters
        '''
        matrix = np.zeros((self.n_cells, self.n_cells))

        for cell1 in pattern.keys():
            for cell2 in pattern.keys():
                x1 = pattern[cell1]
                x2 = pattern[cell2]
                if cell1 != cell2:
                    matrix[cell1, cell2] = self.kernel.kernel(x1, x2)

        return matrix

    def _build_pattern_matrix_symmetric_(self, pattern):
        '''
        Builds the symmetric part of interaction matrix of the specified pattern.
        The pattern-specific symmetric interaction matrices are mostly used in the calculations
        of the order parameters
        '''
        matrix = np.zeros((self.n_cells, self.n_cells))

        for cell1 in pattern.keys():
            for cell2 in pattern.keys():
                x1 = pattern[cell1]
                x2 = pattern[cell2]
                if cell1 != cell2:
                    matrix[cell1, cell2] = self.kernel.symmetric_kernel(x1, x2)

        return matrix

    def _build_autoassociative_matrix_(self):
        '''
        Builds the autoassociative matrix by computing and summing the contribution of each pattern.
        Pattern-specific matrices are also saved in `self.pattern_matrices` and `self.pattern__matrices_symmetric`.
        '''
        self.autoassociative_matrix = np.zeros((self.n_cells, self.n_cells))
        self.pattern_matrices = []
        self.pattern_matrices_symmetric = []
        for pattern in self.patterns:
            pattern_mat = self._build_pattern_matrix_(pattern)
            pattern_symmetric_mat = self._build_pattern_matrix_symmetric_(
                pattern)
            self.autoassociative_matrix += pattern_mat
            self.pattern_matrices.append(pattern_mat)
            self.pattern_matrices_symmetric.append(pattern_symmetric_mat)

        self.autoassociative_matrix = self.autoassociative_matrix / \
            len(self.patterns)

    def _build_heteroassociative_matrix_(self):
        '''
        Builds heteroassociative interaction matrix  by linking together the end of 
        each pattern to the beginning of each pattern that follows the first in any of the chains

        '''
        self.heteroassociative_matrix = np.zeros((self.n_cells, self.n_cells))
        for i, pattern1 in enumerate(self.patterns):
            for j, pattern2 in enumerate(self.patterns):
                if self.chain_transitions[i, j] > 0:
                    for cell1 in pattern1.keys():
                        for cell2 in pattern2.keys():

                            x1 = pattern1[cell1]
                            # offsets so that pattern 2 begins at end of 1
                            x2 = pattern2[cell2]+1

                            self.heteroassociative_matrix[cell1,
                                                          cell2] += self.kernel.kernel(x1, x2)
                            self.heteroassociative_matrix[cell2,
                                                          cell1] += self.kernel.kernel(x2, x1)
        self.heteroassociative_matrix = self.heteroassociative_matrix / \
            len(self.patterns)


# FUNCTIONS

def pattern_overlap_matrix(pattern_structure: PatternStructure) -> np.array:
    '''
    Computes the pattern-pattern overlap for all pattern pairs in the memory
    structure. Overlaps are defined as the number of common cells between the patterns
    over the total number of cells in the each pattern.
    '''
    matrix = np.zeros((pattern_structure.n_patterns,
                       pattern_structure.n_patterns))
    for i in range(pattern_structure.n_patterns):
        for j in range(i+1, pattern_structure.n_patterns):
            cells1 = set([c for c in pattern_structure.patterns[i].keys()])
            cells2 = set([c for c in pattern_structure.patterns[j].keys()])
            intersection = len(cells1.intersection(cells2))

            matrix[i, j] = matrix[j, i] = float(intersection) / \
                float(pattern_structure.cells_per_pattern)

    return matrix


def build_correlated_activity(pattern_struct, pattern_num, position=0.5, sigma=0):
    '''
    Produces an activity configuration correlated with the given pattern in the given position.
    '''

    V = np.zeros(pattern_struct.n_cells)
    noise = np.random.normal(0, sigma, pattern_struct.n_cells)
    pattern = pattern_struct.patterns[pattern_num]
    for cell in pattern.keys():
        V[cell] = pattern_struct.kernel.symmetric_kernel(
            position, pattern[cell])
    return V + noise
