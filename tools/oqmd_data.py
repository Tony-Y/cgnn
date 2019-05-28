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
"""This program writes down OQMD entries in the Materials Project format."""

from qmpy import *
from pymatgen.core.structure import Structure
import numpy as np

from pydash import py_
from tqdm import tqdm

import os

def get_valid_entries(max_fe=5):
    entries = Calculation.objects \
              .filter(label__in=['static', 'standard']) \
              .filter(converged=True) \
              .exclude(formationenergy=None) \
              .exclude(entry__duplicate_of=None) \
              .filter(formationenergy__delta_e__lt=max_fe) \
              .values_list('entry__duplicate_of', flat=True) \
              .distinct()
    return list(entries)

def get_calculations(entry_id, type):
    return Calculation.objects \
          .filter(entry__duplicate_of=entry_id) \
          .filter(converged=True) \
          .exclude(formationenergy=None) \
          .filter(label=type).all()

def get_valid_calculation(entry_id):
    c = get_calculations(entry_id, 'static')
    if len(c) == 0:
        c = get_calculations(entry_id, 'standard')
    imin = 0
    if len(c) > 1:
        imin = np.argmin([x.formationenergy_set.first().delta_e for x in c])
    return c[imin]

def pmg_structure(s, magnetic=False):
    if magnetic:
        sp = {"magmom": s.magmoms}
    else:
        sp = None
    return Structure(lattice=s.cell,
                     species=s.atomic_numbers,
                     coords=s.cartesian_coords,
                     coords_are_cartesian=True,
                     site_properties=sp)

def get_smallest(s):
    p = s.get_primitive_structure()
    while(True):
        t = p.get_primitive_structure()
        if len(t) == len(p):
            return p
        p = t

def get_properties(entry_id):
    c = get_valid_calculation(entry_id)

    structure = pmg_structure(c.output, magnetic=c.magmom is not None)
    prim_structure = get_smallest(structure)

    d = dict()
    d['material_id'] = 'oqmd-{}'.format(entry_id)
    d['nelements'] = c.input.ntypes
    d['nsites'] = len(prim_structure)
    d['structure'] = structure

    try:
        d['spacegroup'] = dict(number=prim_structure.get_space_group_info()[1])
    except TypeError:
        print('spacegroup failed')
        print('entry_id', entry_id)
        d['spacegroup'] = dict(number=-1)

    d['energy_per_atom'] = c.energy_pa
    d['formation_energy_per_atom'] = c.formationenergy_set.first().delta_e

    if c.magmom is None:
        d['total_magnetization'] = 0.0
    else:
        d['total_magnetization'] = abs(c.magmom)

    if c.band_gap is None:
        d['band_gap'] = 0.0
    else:
        d['band_gap'] = c.band_gap

    return d

def main(data_dir, chunk_size=1000):
    if not os.path.isdir(data_dir):
        print('Not found the data directory: {}'.format(data_dir))
        exit(1)

    entries = get_valid_entries()
    print('Total Materials:', len(entries))
    np.random.seed(0)
    np.random.shuffle(entries)
    for n, chunk in enumerate(tqdm(py_.chunk(entries, chunk_size))):
        materials = [get_properties(entry_id) for entry_id in chunk]
        file_path = os.path.join(data_dir,"mp_data_{:03d}.npz".format(n))
        np.savez_compressed(file_path, materials=materials)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="The OQMD Dataset")
    parser.add_argument("--data_dir", metavar='PATH', type=str, default='data',
                        help='The path to a data directory (default: data)')

    options = vars(parser.parse_args())

    main(**options)
