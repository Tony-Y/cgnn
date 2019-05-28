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
"""A crystal graph coordinator for CGNN."""

import numpy as np
import pandas as pd
import pbc
from pymatgen.core.periodic_table import Element
from sklearn.cluster import KMeans
import sys
import os
import glob
from joblib import Parallel, delayed

import warnings

CC_CUTOFF = 0.03
RADIUS_FACTOR = 1.2

def get_nbrs(crystal_xyz, crystal_lat, R_max):
    A = np.transpose(crystal_lat)
    B = np.linalg.inv(A)
    crystal_red = np.matmul(crystal_xyz, np.transpose(B))
    crystal_nbrs = pbc.get_shortest_distances(crystal_red, A, R_max,
                                              crdn_only=True)
    return crystal_nbrs

def get_radius(e):
    r = e.atomic_radius_calculated
    if r is None:
        r = e.atomic_radius
    if r is None:
        raise NameError('Not found the atomic radius for the element: {}'.format(e))
    else:
        return r

def get_nn_cluster(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    cc_distance = abs(kmeans.cluster_centers_[0,0]-kmeans.cluster_centers_[1,0])
    if cc_distance > CC_CUTOFF:
        nnc_index = np.argmin(kmeans.cluster_centers_)
        nnc_nbrs = [i for i in range(len(X)) if kmeans.labels_[i] == nnc_index]
        return np.array(nnc_nbrs)
    else:
        return np.arange(len(X))

def get_nnc_loop(X):
    indices = np.arange(len(X))
    for _ in range(10):
        if len(indices) <= 1:
            break
        nnc = get_nn_cluster(X)
        if len(nnc) == len(indices):
            break
        else:
            indices = indices[nnc]
            X = X[nnc]
    return indices

def get_neighbors(geom):
    elems = [Element.from_Z(z) for z in geom.atomic_numbers]
    radii = np.array([get_radius(e) for e in elems])
    cutoff = radii[:,np.newaxis] + radii[np.newaxis, :]
    vol_atom = (4 * np.pi / 3) * np.array([r**3 for r in radii]).sum()
    factor_vol = (geom.volume / vol_atom)**(1.0/3.0)
    factor = factor_vol * RADIUS_FACTOR
    cutoff *= factor
    candidates = get_nbrs(geom.cart_coords, geom.lattice.matrix, cutoff)
    neighbors = []
    for j in range(len(candidates)):
        dists = []
        for nbr in candidates[j]:
            i = nbr[0]
            d = nbr[1]
            r = nbr[2]
            dists.append(d / cutoff[j,i])
        X = np.array(dists).reshape((-1, 1))
        nnc_nbrs = get_nnc_loop(X)
        neighbors.append([candidates[j][i][0] for i in nnc_nbrs])
    return neighbors

def get_structure(m):
    if m['nsites'] == len(m['structure']):
        return m['structure']
    else:
        s = m['structure'].get_primitive_structure()
        for _ in range(10):
            if m['nsites'] == len(s):
                return s
            else:
                s = s.get_primitive_structure()
        raise NameError('The primitive structure could not be got for {}'.format(m['material_id']))

def load_materials(filepath):
    try:
        data = np.load(filepath)['materials']
    except UnicodeError:
        data = np.load(filepath, encoding='latin1')['materials']
    return data

def process(data_path):
    materials = load_materials(data_path)
    material_ids = [m['material_id'] for m in materials]
    structures = [get_structure(m) for m in materials]
    data_ac = []
    data_nbr = []
    for geom in structures:
        neighbors = get_neighbors(geom)
        data_ac.append(geom.atomic_numbers)
        data_nbr.append(neighbors)
    graph_path = data_path.replace('mp_data', 'mp_graph')
    np.savez_compressed(graph_path, graph_names=material_ids,
                        graph_nodes=data_ac, graph_edges=data_nbr)

def main(data_dir, num_cpus):
    if not os.path.isdir(data_dir):
        print('Not found the data directory: {}'.format(data_dir))
        exit(1)

    data_files = sorted(glob.glob(os.path.join(data_dir, 'mp_data_*.npz')))
    Parallel(n_jobs=num_cpus, verbose=10)([delayed(process)(path) for path in data_files])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crystal Graph Coordinator.')
    parser.add_argument('--data_dir', metavar='PATH', type=str, default='data',
                        help='The path to a data directory (default: data)')
    parser.add_argument('--num_cpus', metavar='N', type=int, default=-1,
                        help='The number of CPUs used for processing (default: -1)')
    options = vars(parser.parse_args())

    main(**options)
