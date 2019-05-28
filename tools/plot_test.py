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
"""A program that plots predictions compared to targets."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def rmse(y_pred, y_true):
    mse = np.mean((y_pred-y_true)**2)
    return  np.sqrt(mse)

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred-y_true))

dx = 0.5
if len(sys.argv) == 2:
    dx = float(sys.argv[1])

df = pd.read_csv("test_predictions.csv")

print('RMSE:', rmse(df.prediction.values, df.target.values))
print('MAE:', mae(df.prediction.values, df.target.values))

max_value = df[['prediction', 'target']].max().max()
print('max_value', max_value)
if max_value > 0:
    max_value = (int(max_value / dx) + 1) * dx
else:
    max_value = int(max_value / dx) * dx

min_value = df[['prediction', 'target']].min().min()
print('min_value', min_value)
if min_value > 0:
    min_value = int(min_value / dx) * dx
else:
    min_value = (int(min_value / dx) - 1) * dx

df.plot.scatter(x="target",y="prediction",
                xlim=(min_value, max_value),
                ylim=(min_value, max_value))

plt.show()
