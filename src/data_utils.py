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
"""Data untilities for CGNN."""

import os.path
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from models import GGNNInput

def load_target(target_name, file_path):
    df = pd.read_csv(file_path)
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

class Graph(object):
    def __init__(self, graph, node_vectors):
        self.nodes, self.neighbors = graph
        self.neighbors = list(self.neighbors)

        n_types = len(node_vectors)
        n_nodes = len(self.nodes)

        # Make node representations
        self.nodes = [node_vectors[i] for i in self.nodes]

        self.nodes = np.array(self.nodes, dtype=np.float32)
        self.edge_sources = np.concatenate([[i] * len(self.neighbors[i]) for i in range(n_nodes)])
        self.edge_targets = np.concatenate(self.neighbors)

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
        self.graph_data = load_graph_data(graph_data_path)
        self.graph_data = [Graph(self.graph_data[name], self.node_vectors)
                           for name in self.graph_names]

    def __getitem__(self, index):
        return self.graph_data[index], self.targets[index]

    def __len__(self):
        return len(self.graph_names)

def graph_collate(batch):
    nodes = []
    edge_sources = []
    edge_targets = []
    graph_indices = []
    node_counts = []
    targets = []
    total_count = 0
    for i, (graph, target) in enumerate(batch):
        nodes.append(graph.nodes)
        edge_sources.append(graph.edge_sources + total_count)
        edge_targets.append(graph.edge_targets + total_count)
        graph_indices += [i] * len(graph)
        node_counts.append(len(graph))
        targets.append(target)
        total_count += len(graph)

    nodes = np.concatenate(nodes)
    edge_sources = np.concatenate(edge_sources)
    edge_targets = np.concatenate(edge_targets)

    input = GGNNInput(nodes, edge_sources, edge_targets, graph_indices, node_counts)
    targets = torch.Tensor(targets)
    return input, targets
