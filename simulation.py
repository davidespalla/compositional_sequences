import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
import pickle

import PatternStructure as ps
import AssociativeNetwork as nt
import functions as fs


# PARAMETERS
OUTPUT_FOLDER_PATH = '../data/QM_phase_plane'

PARAMS = {'GAMMA': 0.5,  # weight of antisymmetric component
          'XI': 0.1,  # spatial scale of the kernel
          'N_CELLS': 1000,  # number of cells in the network
          'CELLS_PER_PATTERN': 100,  # cells active in each patterns
          'N_PATTERNS': [64],  # P
          'N_CHAINS': [1, 2, 4, 8, 16, 32, 64],  # M
          'PATTERNS_PER_CHAIN': [1, 2, 4, 8, 16, 32, 64, 128],  # Q
          'DYNAMIC_SPARSITY': .2,  # sparsity of the dynamics
          'N_STEPS': 200,  # number of dynamic steps
          'N_DRAWS': 2}  # number o


PARAMS['DYNAMIC_SPARSITY'] = PARAMS['DYNAMIC_SPARSITY'] * \
    (PARAMS['CELLS_PER_PATTERN']/PARAMS['N_CELLS'])

Path(OUTPUT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)

# save simualtion parameters
with open(OUTPUT_FOLDER_PATH+'/PARAMS.pickle', 'wb') as handle:
    pickle.dump(PARAMS, handle, protocol=pickle.HIGHEST_PROTOCOL)

for i, (n_patterns, n_chains, patterns_per_chain) in enumerate(product(PARAMS['N_PATTERNS'], PARAMS['N_CHAINS'], PARAMS['PATTERNS_PER_CHAIN'])):
    tot_runs = len(PARAMS['N_PATTERNS']) * \
        len(PARAMS['N_CHAINS'])*len(PARAMS['PATTERNS_PER_CHAIN'])

    print(
        f'Computing step {i+1}/{tot_runs}, P={n_patterns}, M={n_chains},Q={patterns_per_chain}')

    # initialize dict for saving results
    simulation_data = {'M': [], 'Q': [], 'draw': [],
                       'retrieval_quality': [], 'obedience': []}
    for d in range(PARAMS['N_DRAWS']):

        # Memory structure
        kernel = ps.InteractionKernel(gamma=PARAMS['GAMMA'], xi=PARAMS['XI'])
        memories = ps.PatternStructure(n_cells=PARAMS['N_CELLS'],
                                       n_patterns=n_patterns,
                                       cells_per_pattern=PARAMS['CELLS_PER_PATTERN'],
                                       n_chains=n_chains,
                                       patterns_per_chain=patterns_per_chain,
                                       kernel=kernel)
        memories.generate_patterns()
        memories.generate_chains()
        memories.build_interactions()

        # initialize network
        net = nt.Network(J=memories.interaction_matrix,
                         transfer_func=nt.ReLu,
                         dynamic_func=nt.net_dynamics)

        # run dynamics
        starting_map = np.random.choice(n_patterns)
        initial_config = ps.build_correlated_activity(
            memories, starting_map, position=0.1)
        initial_config = initial_config/np.mean(initial_config)
        net.run_dynamics(initial_config, n_steps=PARAMS['N_STEPS'],
                         sparsity=PARAMS['DYNAMIC_SPARSITY'])

        coherence = fs.coherence_timecourse(memories, net.history)
        retrieved_sequence = fs.retrieved_sequence(memories, net.history)

        simulation_data['M'].append(n_chains)
        simulation_data['Q'].append(patterns_per_chain)
        simulation_data['draw'].append(d)
        simulation_data['retrieval_quality'].append(
            fs.compute_retrieval_quality(coherence))
        simulation_data['obedience'].append(fs.compute_obedience(
            retrieved_sequence, memories.chain_transitions))

    simulation_data = pd.DataFrame(simulation_data)
    simulation_data.to_csv(
        OUTPUT_FOLDER_PATH+f'/P{n_patterns}_M{n_chains}_Q{patterns_per_chain}.cvs')

print(f'DONE, output saved @ {OUTPUT_FOLDER_PATH}')
