#   Copyright 2019 Takenori Yamamoto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""A program that compiles the OQMD dataset for CGNN."""

from __future__ import print_function

import numpy as np
import pandas as pd
import json
import sys
import os
import glob
sys.path.append("../tools")
from volume import get_vol

if len(sys.argv) != 2:
    print("Need a path to the directory with OQMD data and graph files as an argumnet.")
    exit()
mp_path = sys.argv[1]
seed = 12345

data_files = sorted(glob.glob(os.path.join(mp_path,'mp_data_*.npz')))
print('Data Files:', len(data_files))

id_key = 'material_id'
st_key = 'structure'
tm_key = 'total_magnetization'
sg_key1 = 'spacegroup'
sg_key2 = 'number'
properties=['nelements',
            'nsites',
            'energy_per_atom',
            'formation_energy_per_atom',
            'band_gap']

def volume_per_atom(x):
    s = x[st_key]
    return s.volume / len(s)

def magnetization_per_atom(x):
    s = x[st_key]
    m = x[tm_key]
    return m / len(s)

def load_materials(filepath):
    try:
        data = np.load(filepath)['materials']
    except UnicodeError:
        data = np.load(filepath, encoding='latin1')['materials']
    return data

def get_prop(path):
    data = load_materials(path)
    print('Data:', len(data), path)
    df = pd.DataFrame()
    df['name'] = [x[id_key] for x in data]
    df['formula'] = [x[st_key].composition.reduced_formula for x in data]
    df[sg_key1] = [x[sg_key1][sg_key2] for x in data]
    for p in properties:
        df[p] = [x[p] for x in data]
    df['volume_per_atom'] = [volume_per_atom(x) for x in data]
    df['magnetization_per_atom'] = [magnetization_per_atom(x) for x in data]
    df['atomic_volume_per_atom'] = [get_vol(x) for x in df.formula]
    df['volume_deviation'] = 1 - df['atomic_volume_per_atom'] / df['volume_per_atom']
    return df

df_targets = pd.concat([get_prop(path) for path in data_files], ignore_index=True)
df_targets.to_csv('targets.csv', index=False)
print('Total Data:', len(df_targets))

graph_files = [x.replace('mp_data_','mp_graph_') for x in data_files]

graph_names = []
graph_nodes = []
graph_edges = []
for path in graph_files:
    data = np.load(path)
    graph_names += list(data['graph_names'])
    graph_nodes += list(data['graph_nodes'])
    graph_edges += list(data['graph_edges'])

unique_z = np.unique(np.concatenate(graph_nodes))
num_z = len(unique_z)
print('unique_z:', num_z)
print('min z:', np.min(unique_z))
print('max z:', np.max(unique_z))
print(unique_z)

z_dict = {z:i for i, z in enumerate(unique_z)}

graphs = dict()
max_neighbors = 0
for name, nodes, neighbors in zip(graph_names, graph_nodes, graph_edges):
    nodes = [z_dict[z] for z in nodes]
    neighbor_counts = [len(nbrs) for nbrs in neighbors]
    max_count = np.max(neighbor_counts)
    max_neighbors = max(max_neighbors, max_count)

    graphs[name] = (nodes, neighbors)
np.savez_compressed("graph_data.npz", graph_dict=graphs)

# Configuration file
config = dict()
config["atomic_numbers"] = unique_z.tolist()
config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot encoding
with open("config.json", 'w') as f:
    json.dump(config, f)

# Split formulas
condition_unary = df_targets.nelements == 1
unary_formulas = df_targets[condition_unary].formula.unique()
multi_formulas = df_targets[~condition_unary].formula.unique()
print('Unary formulas:', len(unary_formulas))
print('Multi formulas:', len(multi_formulas))

np.random.seed(seed)
np.random.shuffle(multi_formulas)
n_test = int(len(multi_formulas) * 0.1)
n_val = n_test * 2
test_formulas = list(multi_formulas[:n_test])
val_formulas = list(multi_formulas[n_test:n_val])
train_formulas = list(multi_formulas[n_val:]) + list(unary_formulas)
print('Train formulas:', len(train_formulas))
print('Val formulas:', len(val_formulas))
print('Test formulas:', len(test_formulas))

train_indices = df_targets[df_targets.formula.isin(train_formulas)].index
val_indices = df_targets[df_targets.formula.isin(val_formulas)].index
test_indices = df_targets[df_targets.formula.isin(test_formulas)].index
print('Train:', len(train_indices))
print('Val:', len(val_indices))
print('Test:', len(test_indices))

# Split file
split = dict()
split["train"] = list(train_indices)
split["val"] = list(val_indices)
split["test"] = list(test_indices)
with open("split.json", 'w') as f:
    json.dump(split, f)
