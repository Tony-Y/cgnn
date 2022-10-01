#   Copyright 2019-2022 Takenori Yamamoto
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
"""Data untilities for CGNN."""

import os.path
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from models import GGNNInput

def load_target(target_name, file_path):
    df = pd.read_csv(file_path, usecols=["name", target_name])
    graph_names = df["name"].values
    targets = df[target_name].values
    return graph_names, targets

def load_graph_data(file_path):
    try:
        graphs = np.load(file_path, allow_pickle=True)['graph_dict'].item()
    except UnicodeError:
        graphs = np.load(file_path, allow_pickle=True, encoding='latin1')['graph_dict'].item()
        graphs = { k.decode() : v for k, v in graphs.items() }
    return graphs

def get_num_edge_labels(graphs):
    graph = next(iter(graphs.values()))
    if len(graph) == 2: # Version 1
        _, neighbors = graph
        if isinstance(neighbors, tuple):
            return len(neighbors)
        else:
            return 1
    else: # Version 2
        type_counts = graph[1]
        return len(type_counts)

class Graph(object):
    __slots__ = ['nodes', 'edge_sources', 'edge_targets']
    def __init__(self, graph, num_edge_labels):
        if len(graph) == 2:
            self._process_v1(graph, num_edge_labels)
        else:
            self._process_v2(graph, num_edge_labels)
    
    def _process_v1(self, graph, num_edge_labels):
        self.nodes, neighbors = graph
        if num_edge_labels > 1:
            neighbors = [list(neighbors[i]) for i in range(num_edge_labels)]
        else:
            neighbors = [list(neighbors)]

        self.nodes = np.array(self.nodes, dtype=int)

        n_nodes = len(self.nodes)
        dtype = np.int64
        self.edge_sources, self.edge_targets = [], []
        for nbrs in neighbors:
            nbrs = [np.zeros(0, dtype=dtype) if x is None else np.array(x, dtype=dtype) for x in nbrs]
            self.edge_sources.append(
                    np.concatenate([np.array([i] * len(nbrs[i]), dtype=dtype) for i in range(n_nodes)])
                )
            self.edge_targets.append(np.concatenate(nbrs))
    
    def _process_v2(self, graph, num_edge_labels):
        self.nodes, type_counts, neighbor_counts, neighbors = graph
        neighbor_counts = neighbor_counts.reshape(num_edge_labels,-1)
        
        self.nodes = np.array(self.nodes, dtype=int)

        n_nodes = len(self.nodes)
        dtype = np.int64
        
        self.edge_sources= []
        for nbc in neighbor_counts:
            src = [np.full(c, node, dtype=dtype) for node, c in enumerate(nbc)]
            self.edge_sources.append(np.concatenate(src))
        
        self.edge_targets = []
        s = 0
        for c in type_counts:
            e = s + c
            tgt = neighbors[s:e].astype(dtype)
            self.edge_targets.append(tgt)
            s = e

    def __len__(self):
        return len(self.nodes)

class GraphDataset(Dataset):
    def __init__(self, path, target_name):
        super(GraphDataset, self).__init__()

        self.path = path
        target_path = os.path.join(path, "targets.csv")
        self.graph_names, self.targets = load_target(target_name, target_path)
        config_path = os.path.join(path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        self.node_vectors = config["node_vectors"]
        graph_data_path = os.path.join(path, "graph_data.npz")
        graph_data = load_graph_data(graph_data_path)
        self.num_edge_labels = get_num_edge_labels(graph_data)
        self.graph_data = []
        for name in self.graph_names:
            graph = Graph(graph_data[name], self.num_edge_labels)
            self.graph_data.append(graph)
            del graph_data[name]

    def __getitem__(self, index):
        return self.graph_data[index], self.targets[index]

    def __len__(self):
        return len(self.graph_names)

def graph_collate(batch, num_edge_labels):
    nodes = []
    edge_sources = [[] for _ in range(num_edge_labels)]
    edge_targets = [[] for _ in range(num_edge_labels)]
    graph_indices = []
    node_counts = []
    targets = []
    total_count = 0
    for i, (graph, target) in enumerate(batch):
        nodes.append(graph.nodes)
        for j in range(num_edge_labels):
            edge_sources[j].append(graph.edge_sources[j] + total_count)
            edge_targets[j].append(graph.edge_targets[j] + total_count)
        graph_indices += [i] * len(graph)
        node_counts.append(len(graph))
        targets.append(target)
        total_count += len(graph)

    nodes = np.concatenate(nodes)
    for j in range(num_edge_labels):
        edge_sources[j] = np.concatenate(edge_sources[j])
        edge_targets[j] = np.concatenate(edge_targets[j])

    input = GGNNInput(nodes, edge_sources, edge_targets, graph_indices, node_counts)
    targets = torch.tensor(targets, dtype=torch.float32)
    return input, targets
