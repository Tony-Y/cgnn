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
"""The main program of CGNN."""

import time
import json
import os
import copy

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

import pytorch_warmup as warmup

from models import GGNN
from model_utils import Model
from data_utils import GraphDataset, graph_collate

def use_setpLR(param):
    ms = param["milestones"]
    return ms[0] < 0

def create_model(device, model_param, optimizer_param, scheduler_param, summary_writer, sw_step_interval):
    model = GGNN(**model_param).to(device)
    clip_value = optimizer_param.pop("clip_value")
    optim_name = optimizer_param.pop("optim")
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), momentum=0.9,
                              nesterov=True, **optimizer_param)
    elif optim_name == "adam":
        optimizer = optim.AdamW(model.parameters(), **optimizer_param)
    elif optim_name == "amsgrad":
        optimizer = optim.AdamW(model.parameters(), amsgrad=True,
                               **optimizer_param)
    else:
        raise NameError("optimizer {} is not supported".format(optim_name))
    use_cosine_annealing = scheduler_param.pop("cosine_annealing")
    use_warmup = scheduler_param.pop("warmup")
    num_warmup_steps = scheduler_param.pop("num_warmup_steps")
    if use_cosine_annealing:
        params = dict(T_max=scheduler_param["milestones"][0],
                      eta_min=scheduler_param["gamma"])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        print('CosineAnnealingLR eta_min:', lr_scheduler.eta_min, 'T_max:', lr_scheduler.T_max)
    elif use_setpLR(scheduler_param):
        scheduler_param["step_size"] = abs(scheduler_param["milestones"][0])
        scheduler_param.pop("milestones")
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)
        print('StepLR step_size:', lr_scheduler.step_size)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
        print('MultiStepLR milestones:', lr_scheduler.milestones)
    if use_warmup:
        if num_warmup_steps == 0 and optim_name in ["adam", "amsgrad"]:
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        else:
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=num_warmup_steps)
    else:
        warmup_scheduler = None
    return Model(device, model, optimizer, lr_scheduler, warmup_scheduler,
                 clip_value, summary_writer, sw_step_interval)

def main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         num_epochs, seed, load_model, summary_writer, sw_step_interval):
    print("Seed:", seed)
    print()
    torch.manual_seed(seed)

    # Create dataset
    split_file_path = dataset_param.pop("split_file")
    dataset_param["path"] = dataset_param.pop("dataset_path")
    dataset = GraphDataset(**dataset_param)
    print("Number of dege labels:", dataset.num_edge_labels)
    dataloader_param["collate_fn"] = lambda batch: graph_collate(batch, dataset.num_edge_labels)
    model_param["n_edge_labels"] = dataset.num_edge_labels
    model_param["node_vectors"] = dataset.node_vectors

    # split the dataset into training, validation, and test sets.
    if split_file_path is not None and os.path.isfile(split_file_path):
        with open(split_file_path) as f:
            split = json.load(f)
    else:
        print("No split file. Default split: 256 (train), 32 (val), 32 (test)")
        split = {"train": range(256), "val": range(256, 288), "test": range(288, 320)}
    print(" ".join(["{}: {}".format(k, len(x)) for k, x in split.items()]))

    train_sampler = SubsetRandomSampler(split["train"])
    val_sampler = SubsetRandomSampler(split["val"])
    train_dl = DataLoader(dataset, sampler=train_sampler, **dataloader_param)
    val_dl = DataLoader(dataset, sampler=val_sampler, **dataloader_param)

    batches_per_epoch = len(train_dl)
    print('batches per epoch:', batches_per_epoch)
    scheduler_param["milestones"] = [n * batches_per_epoch for n in scheduler_param["milestones"]]

    # Create a CGNN model
    model = create_model(device, model_param, optimizer_param, scheduler_param, summary_writer, sw_step_interval)
    if load_model:
        print("Loading weights from model.pth")
        print()
        model.load()
    #print("Model:", model.device)
    if summary_writer:
        print("Summary Writer will be used. You can start TensorBoard through " \
              "the command line: tensorboard --logdir=runs")
        print(f"SW step interval: {sw_step_interval}")
        print()

    # Train
    model.train(train_dl, val_dl, num_epochs)
    if num_epochs > 0:
        model.save()

    # Test
    test_set = Subset(dataset, split["test"])
    test_dl = DataLoader(test_set, **dataloader_param)
    outputs, targets = model.evaluate(test_dl)
    names = [dataset.graph_names[i] for i in split["test"]]
    df_predictions = pd.DataFrame({"name": names, "prediction": outputs, "target": targets})
    df_predictions.to_csv("test_predictions.csv", index=False)

    print("\nEND")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Crystal Graph Neural Networks")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_node_feat", type=int, default=4)
    parser.add_argument("--n_hidden_feat", type=int, default=16)
    parser.add_argument("--n_graph_feat", type=int, default=32)
    parser.add_argument("--n_conv", type=int, default=3)
    parser.add_argument("--n_fc", type=int, default=2)
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--use_batch_norm", action='store_true')
    parser.add_argument("--node_activation", type=str, default="None")
    parser.add_argument("--use_node_batch_norm", action='store_true')
    parser.add_argument("--edge_activation", type=str, default="None")
    parser.add_argument("--use_edge_batch_norm", action='store_true')
    parser.add_argument("--n_edge_net_feat", type=int, default=16)
    parser.add_argument("--n_edge_net_layers", type=int, default=0)
    parser.add_argument("--edge_net_activation", type=str, default="elu")
    parser.add_argument("--use_edge_net_batch_norm", action='store_true')
    parser.add_argument("--use_fast_edge_network", action='store_true')
    parser.add_argument("--fast_edge_network_type", type=int, default=0)
    parser.add_argument("--use_aggregated_edge_network", action='store_true')
    parser.add_argument("--edge_net_cardinality", type=int, default=32)
    parser.add_argument("--edge_net_width", type=int, default=4)
    parser.add_argument("--use_edge_net_shortcut", action='store_true')
    parser.add_argument("--n_postconv_net_layers", type=int, default=0)
    parser.add_argument("--postconv_net_activation", type=str, default="elu")
    parser.add_argument("--use_postconv_net_batch_norm", action='store_true')
    parser.add_argument("--conv_bias", action='store_true')
    parser.add_argument("--edge_net_bias", action='store_true')
    parser.add_argument("--postconv_net_bias", action='store_true')
    parser.add_argument("--full_pooling", action='store_true')
    parser.add_argument("--gated_pooling", action='store_true')
    parser.add_argument("--conv_type", type=int, default=0)
    parser.add_argument("--conv_labels", nargs='+', type=int, default=[])
    parser.add_argument("--output_activation", type=str, default="None")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--milestones", nargs='+', type=int, default=[10])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--cosine_annealing", action='store_true')
    parser.add_argument("--warmup", action='store_true')
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--dataset_path", type=str, default="datasets/Nomad2018")
    parser.add_argument("--target_name", type=str, default="formation_energy_ev_natom")
    parser.add_argument("--split_file", type=str, default="datasets/Nomad2018/split.json")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--use_extension", action='store_true')
    parser.add_argument("--summary_writer", action='store_true')
    parser.add_argument("--sw_step_interval", type=int, default=20)
    options = vars(parser.parse_args())

    if not torch.cuda.is_available():
        options["device"] = "cpu"
    print("Device:", options["device"])
    print()
    device = torch.device(options["device"])

    # Model parameters
    model_param_names = [
        "n_node_feat", "n_hidden_feat", "n_graph_feat", "n_conv", "n_fc",
        "activation", "use_batch_norm", "node_activation", "use_node_batch_norm",
        "edge_activation", "use_edge_batch_norm", "n_edge_net_feat",
        "n_edge_net_layers", "edge_net_activation", "use_edge_net_batch_norm",
        "use_fast_edge_network", "fast_edge_network_type",
        "use_aggregated_edge_network", "edge_net_cardinality", "edge_net_width",
        "use_edge_net_shortcut", "n_postconv_net_layers", "postconv_net_activation",
        "use_postconv_net_batch_norm", "conv_bias", "edge_net_bias", "postconv_net_bias",
        "full_pooling", "gated_pooling", "conv_type", "conv_labels", "output_activation",
        "use_extension"]
    model_param = {k : options[k] for k in model_param_names if options[k] is not None}
    if model_param["node_activation"].lower() == 'none':
        model_param["node_activation"] = None
    if model_param["edge_activation"].lower() == 'none':
        model_param["edge_activation"] = None
    if model_param["output_activation"].lower() == 'none':
        model_param["output_activation"] = None
    print("Model:", model_param)
    print()

    # Optimizer parameters
    optimizer_param_names = ["optim", "lr", "weight_decay", "clip_value"]
    optimizer_param = {k : options[k] for k in optimizer_param_names if options[k] is not None}
    if optimizer_param["clip_value"] == 0.0:
        optimizer_param["clip_value"] = None
    print("Optimizer:", optimizer_param)
    print()

    # Scheduler parameters
    scheduler_param_names = ["milestones", "gamma", "cosine_annealing", "warmup", "num_warmup_steps"]
    scheduler_param = {k : options[k] for k in scheduler_param_names if options[k] is not None}
    print("Scheduler:", scheduler_param)
    print()

    # Dataset parameters
    dataset_param_names = ["dataset_path", "target_name", "split_file"]
    dataset_param = {k : options[k] for k in dataset_param_names if options[k] is not None}
    print("Dataset:", dataset_param)
    print()

    # Dataloader parameters
    dataloader_param_names = ["num_workers", "batch_size"]
    dataloader_param = {k : options[k] for k in dataloader_param_names if options[k] is not None}
    print("Dataloader:", dataloader_param)
    print()

    main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         options["num_epochs"], options["seed"], options["load_model"], options["summary_writer"],
         options["sw_step_interval"])
