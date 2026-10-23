"""Aligned invariant geometry; no labels, PDB IDs or experimental-method features."""
from __future__ import annotations
import numpy as np

NODE_DIM = 11
EDGE_DIM = 28

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def dihedral(a, b, c, d):
    axis = unit(c-b)
    v, w = a-b, d-c
    v, w = v-(v*axis).sum()*axis, w-(w*axis).sum()*axis
    return np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))

def chain_geometry(atoms, present, mismatches, max_neighbors=32, max_distance=16.):
    """atoms[L,N/CA/C,3]. Nodes preserve full-sequence positions, first model only."""
    from scipy.spatial import cKDTree
    length = len(atoms)
    ca = present[:, 1] & ~mismatches
    positions = np.flatnonzero(ca)
    if not len(positions):
        raise ValueError('Chain has no aligned usable CA positions')
    bb = present.all(1) & ~mismatches
    continuity = np.zeros(max(length-1, 0), bool)
    if length > 1:
        distances = np.linalg.norm(atoms[1:, 1]-atoms[:-1, 1], axis=1)
        continuity = ca[1:] & ca[:-1] & (distances >= 2.8) & (distances <= 4.5)
    peptide = continuity.copy()
    if length > 1:
        cn_distance = np.linalg.norm(atoms[:-1, 2]-atoms[1:, 0], axis=1)
        peptide &= present[:-1, 2] & present[1:, 0] & (cn_distance >= 1.) & (cn_distance <= 1.8)
    values = np.zeros((length, NODE_DIM), np.float32)
    for i in positions:
        previous = i > 0 and peptide[i-1] and bb[i-1] and bb[i]
        following = i+1 < length and peptide[i] and bb[i] and bb[i+1]
        angles = []
        if previous:
            angles.append((0, dihedral(atoms[i-1, 2], *atoms[i])))
            angles.append((2, dihedral(atoms[i-1, 1], atoms[i-1, 2], atoms[i, 0], atoms[i, 1])))
        if following:
            angles.append((1, dihedral(*atoms[i], atoms[i+1, 0])))
        for slot, angle in angles:
            values[i, 2*slot:2*slot+2] = np.sin(angle), np.cos(angle)
            values[i, 6+slot] = 1.
        if i > 0 and i+1 < length and continuity[i-1] and continuity[i]:
            values[i, 9] = np.dot(unit(atoms[i-1, 1]-atoms[i, 1]), unit(atoms[i+1, 1]-atoms[i, 1]))
            values[i, 10] = 1.
    # Orthonormal right-handed backbone frames. Invalid frames stay zero.
    frames = np.zeros((length, 3, 3), np.float32)
    for i in np.flatnonzero(bb & ca):
        first = unit(atoms[i, 2]-atoms[i, 1])
        second = atoms[i, 0]-atoms[i, 1]
        second = unit(second-np.dot(second, first)*first)
        third = np.cross(first, second)
        if np.linalg.norm(third) > .9:
            frames[i] = np.stack((first, second, third))
    frame_valid = np.linalg.norm(frames[:, 2], axis=1) > .9
    xyz = atoms[positions, 1]
    distances, neighbors = cKDTree(xyz).query(xyz, k=min(max_neighbors+1, len(xyz)))
    if distances.ndim == 1:
        distances, neighbors = distances[:, None], neighbors[:, None]
    source, target, ranks, attrs = [], [], [], []
    centers = np.linspace(0., 20., 16)
    for target_index, (ds, ns) in enumerate(zip(distances, neighbors)):
        candidates = [(float(d), int(n)) for d,n in zip(ds, ns) if n != target_index and d <= max_distance]
        candidates.sort(key=lambda pair: (round(pair[0], 5), int(positions[pair[1]])))
        for rank, (distance, source_index) in enumerate(candidates[:max_neighbors], 1):
            i, j = int(positions[target_index]), int(positions[source_index])
            direction = unit(xyz[source_index]-xyz[target_index])
            displacement = j-i
            sequential = abs(displacement)==1 and continuity[min(i,j)]
            edge = np.r_[np.exp(-((distance-centers)/1.25)**2),
                         np.tanh(displacement/32.), np.sign(displacement)*np.log1p(abs(displacement))/8.,
                         float(sequential), distance/20., frames[i]@direction, frames[j]@(-direction),
                         float(frame_valid[i]), float(frame_valid[j])]
            assert len(edge)==EDGE_DIM
            source.append(source_index); target.append(target_index); ranks.append(rank); attrs.append(edge)
    return {'positions':positions.astype(np.int32), 'node':values[positions],
            'edge_index':np.asarray([source,target],np.int32).reshape(2,-1),
            'edge':np.asarray(attrs,np.float32).reshape(-1,EDGE_DIM),
            'edge_rank':np.asarray(ranks,np.int16),
            'continuous':continuity, 'ca_mask':ca, 'backbone_mask':bb}
