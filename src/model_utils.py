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
"""Model untilities for CGNN."""

import copy
import time

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_

class Metric(object):
    def __init__(self, metric_fn, name):
        self.metric_fn = metric_fn
        self.name = name
        self.total_metric = 0.0
        self.total_count = 0

    def __call__(self, predictoins, targets):
        return self.metric_fn(predictoins, targets)

    def add_batch_metric(self, predictoins, targets):
        metric_tensor = self.metric_fn(predictoins, targets)
        self.total_metric += metric_tensor.item() * targets.size(0)
        self.total_count += targets.size(0)
        return metric_tensor

    def get_total_metric(self):
        score = self.total_metric / self.total_count
        self.total_metric = 0.0
        self.total_count = 0
        return score

class History(object):
    def __init__(self,file_path="history.csv"):
        self.history_path = file_path
        self.file = open(file_path, 'w')
        self.first = True

    def write(self, epoch, metrics):
        if self.first:
            self.first = False
            header = ','.join([name for name, _ in metrics])
            header = 'epoch,' + header + '\n'
            self.file.write(header)
        row = ','.join(['{}'.format(metric) for _, metric in metrics])
        row = '{},'.format(epoch) + row + '\n'
        self.file.write(row)
        self.file.flush()

    def close(self):
        self.file.close()

class Checkpoint(object):
    def __init__(self, model):
        self.model = model
        self.best_metric = None
        self.best_weights = model.weights

    def check(self, metric):
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            self.best_weights = self.model.weights

    def restore(self):
        self.model.weights = self.best_weights

class Model(object):
    def __init__(self, device, model, optimizer, scheduler, clip_value=None,
                 metrics=[('loss', nn.MSELoss()), ('mae', nn.L1Loss())]):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = [Metric(metric, name) for name, metric in metrics]
        self.device = device
        self.clip_value = clip_value

    def _set_mode(self, phase):
        if phase == 'train':
            self.model.train()  # Set model to training mode
        else:
            self.model.eval()   # Set model to evaluate mode

    def _process_batch(self, input, targets, phase):
        input = input.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(input)
            metric_tensors = [metric.add_batch_metric(outputs, targets) for metric in self.metrics]

            # backward + optimize only if in training phase
            if phase == 'train':
                loss = metric_tensors[0]
                loss.backward()
                if self.clip_value is not None:
                    clip_grad_value_(self.model.parameters(), self.clip_value)
                self.optimizer.step()

        return metric_tensors, outputs

    def train(self, train_dl, val_dl, num_epochs):
        since = time.time()

        dataloaders = {'train': train_dl, 'val': val_dl}
        history = History()
        checkpoint = Checkpoint(self)
        for epoch in range(num_epochs):
            epoch_since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
            print('current lr:', self.optimizer.param_groups[0]['lr'])

            train_val_metrics = []
            for phase in ['train', 'val']:
                self._set_mode(phase)

                # Iterate over data.
                ##print('batch count:',len(dataloaders[phase]))
                ##nb = 0
                for input, targets in dataloaders[phase]:
                    ##nb += 1
                    ##if (nb % 100) == 0: print(nb)
                    _, outputs = self._process_batch(input, targets, phase)

                epoch_metrics = [(metric.name, metric.get_total_metric()) for metric in self.metrics]
                text = ' '.join(['{}: {:.4f}'.format(name, metric) for name, metric in epoch_metrics])
                print('{} {}'.format(phase, text))

                if phase == 'val':
                    metric = epoch_metrics[1][1]
                    checkpoint.check(metric)
                train_val_metrics += [('_'.join([phase, name]), metric) for name, metric in epoch_metrics]
            history.write(epoch, train_val_metrics)
            self.scheduler.step()
            time_elapsed = time.time() - epoch_since
            print('Elapsed time (sec.): {:.3f}'.format(time_elapsed))
            print()
        history.close()

        if num_epochs > 0:
            time_elapsed = time.time() - since
            print('Total elapsed time (sec.): {:.3f}'.format(time_elapsed))
            print('The best val metric: {:4f}'.format(checkpoint.best_metric))
            print()

            # load the best model weights
            checkpoint.restore()

    def predict(self, dataloader):
        self.model.eval()   # Set model to evaluate mode

        # Iterate over data.
        all_outputs = []
        for input, _ in dataloader:
            input = input.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(input)

            outputs = outputs.to(torch.device("cpu")).numpy()
            all_outputs.append(outputs)

        all_outputs = np.concatenate(all_outputs)
        return all_outputs

    def evaluate(self, dataloader):
        self.model.eval()   # Set model to evaluate mode

        # Iterate over data.
        all_outputs = []
        all_targets = []
        for input, targets in dataloader:
            input = input.to(self.device)
            targets = targets.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(input)

            outputs = outputs.to(torch.device("cpu")).numpy()
            targets = targets.to(torch.device("cpu")).numpy()
            all_outputs.append(outputs)
            all_targets.append(targets)

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        all_outputs = torch.FloatTensor(all_outputs).to(self.device)
        all_targets = torch.FloatTensor(all_targets).to(self.device)

        total_metrics = [(metric.name, metric(all_outputs, all_targets).item()) for metric in self.metrics]
        text = ' '.join(['{}: {:.4f}'.format(name, metric) for name, metric in total_metrics])
        print('test {}'.format(text))

        all_outputs = all_outputs.to(torch.device("cpu")).numpy()
        all_targets = all_targets.to(torch.device("cpu")).numpy()

        return all_outputs, all_targets

    def save(self, model_path="model.pth"):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path="model.pth"):
        self.model.load_state_dict(torch.load(model_path))

    @property
    def weights(self):
        return copy.deepcopy(self.model.state_dict())

    @weights.setter
    def weights(self, state):
        self.model.load_state_dict(state)
