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
"""A function that calculates the total atomic volume for a composition given."""

import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

def get_radius(e):
    r = e.atomic_radius_calculated
    if r is None:
        r = e.atomic_radius
    return r

fpi = 4 * np.pi / 3
vol_dict = dict()
for z in range(1,95):
    e = Element.from_Z(z)
    r = get_radius(e)
    if r is not None:
        vol_dict[z] = fpi * r**3

def get_vol(formula):
    comp = Composition(formula)
    v = np.sum([vol_dict[e.number] * a for e, a in comp.items()])
    v /= comp.num_atoms
    return v
