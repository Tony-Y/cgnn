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
"""A function that calculates distances between every atom and
   its near neighbors in a crystalline material.
"""

import numpy as np

def get_shortest_distances(reduced_coords, amat, R_max, crdn_only=False):
    lmn = []
    for l in range(-1, 2):
        for m in range(-1, 2):
            for n in range(-1, 2):
                lmn.append(np.array([l, m, n]))
    lmn = np.array(lmn)

    rij = reduced_coords[:,np.newaxis] - reduced_coords[np.newaxis,:]
    rij = rij[:, :, np.newaxis, :] + lmn[np.newaxis, np.newaxis]
    Rij = np.matmul(rij, np.transpose(amat))
    rij = np.linalg.norm(Rij, axis=3)
    ijk_crdn = np.where((rij < R_max[:,:,np.newaxis]) & (rij > 1e-8))

    crdn = []
    for n in range(len(reduced_coords)):
        crdn.append([])
    for n in range(len(ijk_crdn[0])):
        i, j, k = ijk_crdn[0][n], ijk_crdn[1][n], ijk_crdn[2][n]
        crdn[j].append([i, rij[i,j,k], Rij[i,j,k]])

    if crdn_only:
        return crdn
    else:
        ij_min = np.argmin(rij, axis=2)
        rij = [ [rij[i, j, ij_min[i,j]] for j in range(len(ij_min))] for i in range(len(ij_min)) ]
        rij = np.array(rij)
        Rij = [ [Rij[i, j, ij_min[i,j]] for j in range(len(ij_min))] for i in range(len(ij_min)) ]
        Rij = np.array(Rij)
        return rij, Rij, crdn
