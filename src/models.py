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
"""Definitions of Neural Network Models for CGNN."""

import torch
import torch.nn as nn
from layers import ( NodeEmbedding, GatedGraphConvolution, GraphPooling,
                     FullConnection, LinearRegression, Extension,
                     get_activation,
                     EdgeNetwork, FastEdgeNetwork, AggregatedEdgeNetwork,
                     LinearPooling, GatedPooling, PostconvolutionNetwork )

class GGNNInput(object):
    def __init__(self, nodes, edge_sources, edge_targets, graph_indices, node_counts):
        self.nodes = torch.tensor(nodes, dtype=torch.int64)
        self.edge_sources = [torch.tensor(x, dtype=torch.int64) for x in edge_sources]
        self.edge_targets = [torch.tensor(x, dtype=torch.int64) for x in edge_targets]
        self.graph_indices = torch.tensor(graph_indices, dtype=torch.int64)
        self.node_counts = torch.tensor(node_counts, dtype=torch.float32)

    def __len__(self):
        return self.nodes.size(0)

    def to(self, device):
        self.nodes = self.nodes.to(device)
        self.edge_sources = [x.to(device) for x in self.edge_sources]
        self.edge_targets = [x.to(device) for x in self.edge_targets]
        self.graph_indices = self.graph_indices.to(device)
        self.node_counts = self.node_counts.to(device)

        return self

class GGNN(nn.Module):
    """
    Gated Graph Neural Networks

    Nodes -> Embedding -> Gated Convolutions -> Graph Pooling -> Full Connections -> Linear Regression

    * Yujia Li, et al., "Gated graph sequence neural networks", https://arxiv.org/abs/1511.05493
    * Justin Gilmer, et al., "Neural Message Passing for Quantum Chemistry", https://arxiv.org/abs/1704.01212
    * Tian Xie, et al., "Crystal Graph Convolutional Neural Networks for an Accurate and
                         Interpretable Prediction of Material Properties", https://arxiv.org/abs/1710.10324
    """
    def __init__(self, n_node_feat, n_edge_labels, n_hidden_feat, n_graph_feat, n_conv, n_fc,
                 activation, use_batch_norm, node_activation, use_node_batch_norm,
                 edge_activation, use_edge_batch_norm, n_edge_net_feat, n_edge_net_layers,
                 edge_net_activation, use_edge_net_batch_norm, use_fast_edge_network,
                 fast_edge_network_type, use_aggregated_edge_network, edge_net_cardinality,
                 edge_net_width, use_edge_net_shortcut, n_postconv_net_layers,
                 postconv_net_activation, use_postconv_net_batch_norm, conv_type, conv_labels,
                 output_activation, node_vectors,
                 conv_bias=False, edge_net_bias=False, postconv_net_bias=False,
                 full_pooling=False, gated_pooling=False,
                 use_extension=False):
        super(GGNN, self).__init__()

        self.n_edge_labels = n_edge_labels
        if len(conv_labels) > 0:
            self.conv_labels = conv_labels
            n_conv = len(self.conv_labels)
        else:
            n_conv *= self.n_edge_labels
            self.conv_labels = [i % self.n_edge_labels for i in range(n_conv)]
        print('n_conv:', n_conv)
        print('conv labels:', self.conv_labels)

        act_fn = get_activation(activation)

        if node_activation is not None:
            node_act_fn = get_activation(node_activation)
        else:
            node_act_fn = None

        if edge_activation is not None:
            edge_act_fn = get_activation(edge_activation)
        else:
            edge_act_fn = None

        if output_activation is not None:
            self.output_act_fn = get_activation(output_activation)
        else:
            self.output_act_fn = None

        postconv_net_act_fn = get_activation(postconv_net_activation)

        if n_edge_net_layers < 1:
            edge_nets = [None for i in range(n_conv)]
        else:
            edge_net_act_fn = get_activation(edge_net_activation)
            if use_aggregated_edge_network:
                edge_nets = [AggregatedEdgeNetwork(n_hidden_feat, n_edge_net_feat,
                             n_edge_net_layers, cardinality=edge_net_cardinality,
                             width=edge_net_width, activation=edge_net_act_fn,
                             use_batch_norm=use_edge_net_batch_norm,
                             bias=edge_net_bias,
                             use_shortcut=use_edge_net_shortcut)
                             for i in range(n_conv)]
            elif use_fast_edge_network:
                edge_nets = [FastEdgeNetwork(n_hidden_feat, n_edge_net_feat,
                             n_edge_net_layers, activation=edge_net_act_fn,
                             net_type=fast_edge_network_type,
                             use_batch_norm=use_edge_net_batch_norm,
                             bias=edge_net_bias,
                             use_shortcut=use_edge_net_shortcut)
                             for i in range(n_conv)]
            else:
                edge_nets = [EdgeNetwork(n_hidden_feat, n_edge_net_feat,
                             n_edge_net_layers, activation=edge_net_act_fn,
                             use_batch_norm=use_edge_net_batch_norm,
                             bias=edge_net_bias,
                             use_shortcut=use_edge_net_shortcut)
                             for i in range(n_conv)]

        if n_postconv_net_layers < 1:
            postconv_nets = [None for i in range(n_conv)]
        else:
            postconv_nets = [PostconvolutionNetwork(n_hidden_feat, n_hidden_feat,
                             n_postconv_net_layers,
                             activation=postconv_net_act_fn,
                             use_batch_norm=use_postconv_net_batch_norm,
                             bias=postconv_net_bias)
                             for i in range(n_conv)]

        self.embedding = NodeEmbedding(n_node_feat, n_hidden_feat, node_vectors)
        self.convs = [GatedGraphConvolution(n_hidden_feat, n_hidden_feat,
                      node_activation=node_act_fn,
                      edge_activation=edge_act_fn,
                      use_node_batch_norm=use_node_batch_norm,
                      use_edge_batch_norm=use_edge_batch_norm,
                      bias=conv_bias,
                      conv_type=conv_type,
                      edge_network=edge_nets[i],
                      postconv_network=postconv_nets[i])
                      for i in range(n_conv)]
        self.convs = nn.ModuleList(self.convs)
        if full_pooling:
            n_steps = n_conv
            if gated_pooling:
                self.pre_poolings = [GatedPooling(n_hidden_feat)
                                     for _ in range(n_conv)]
            else:
                self.pre_poolings = [LinearPooling(n_hidden_feat)
                                     for _ in range(n_conv)]
        else:
            n_steps = 1
            self.pre_poolings = [None for _ in range(n_conv-1)]
            if gated_pooling:
                self.pre_poolings.append(GatedPooling(n_hidden_feat))
            else:
                self.pre_poolings.append(LinearPooling(n_hidden_feat))
        self.pre_poolings = nn.ModuleList(self.pre_poolings)
        self.pooling = GraphPooling(n_hidden_feat, n_steps, activation=act_fn,
                                    use_batch_norm=use_batch_norm)
        self.fcs = [FullConnection(n_hidden_feat, n_graph_feat,
                    activation=act_fn, use_batch_norm=use_batch_norm)]
        self.fcs += [FullConnection(n_graph_feat, n_graph_feat,
                     activation=act_fn, use_batch_norm=use_batch_norm)
                     for i in range(n_fc-1)]
        self.fcs = nn.ModuleList(self.fcs)
        self.regression = LinearRegression(n_graph_feat)
        if use_extension:
            self.extension = Extension()
        else:
            self.extension = None

    def forward(self, input):
        x = self.embedding(input.nodes)
        y = []
        for conv, label, pre_pooling in zip(self.convs, self.conv_labels, self.pre_poolings):
            x = conv(x, input.edge_sources[label], input.edge_targets[label])
            if pre_pooling is not None:
                y.append(pre_pooling(x, input.graph_indices, input.node_counts))
        x = self.pooling(y)
        for fc in self.fcs:
            x = fc(x)
        x = self.regression(x)
        if self.output_act_fn is not None:
            x = self.output_act_fn(x)
        if self.extension is not None:
            x = self.extension(x, input.node_counts)
        return x
