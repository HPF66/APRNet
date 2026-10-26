"""Residue geometry and spatial regional readout with chain-instance preservation."""
import math
import torch
from torch import nn
from protein.utils.checkpoint import EDGE_DIM, NODE_DIM
from protein.model.sequence_context import ResidueContextBackbone

class GeometryMessageBlock(nn.Module):

    def __init__(self, width, dropout):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width * 2 + EDGE_DIM, width), nn.GELU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, edges, attrs):
        source, target = edges
        message = self.message(torch.cat((h[source], h[target], attrs), dim=-1))
        aggregate = torch.zeros_like(h).index_add(0, target, message)
        degree = torch.bincount(target, minlength=len(h)).to(h.dtype).clamp_min(1)
        return self.norm(h + self.update(torch.cat((h, aggregate / degree[:, None]), dim=-1)))

class PDBResidueClassifier(ResidueContextBackbone):
    """ESM-2 650M + HAPR + aligned PDB context; no baseline-mode switches."""

    def __init__(self, hidden, heads, layers, dropout, structure):
        super().__init__(hidden, heads, layers, dropout)
        width = structure['width']
        self.geometry_input = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width))
        self.geometry_node = nn.Linear(NODE_DIM, width)
        self.geometry_blocks = nn.ModuleList((GeometryMessageBlock(width, structure['dropout']) for _ in range(structure['layers'])))
        self.geometry_output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden))
        self.geometry_node_gate = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden))
        self.geometry_region_score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.geometry_readout = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Dropout(structure['dropout']))
        self.geometry_instance_score = nn.Linear(hidden, 1)
        self.geometry_readout_gate = nn.Parameter(torch.tensor(math.log(structure['gate_init'] / (1 - structure['gate_init']))))
        nn.init.zeros_(self.geometry_node_gate[-1].weight)
        nn.init.constant_(self.geometry_node_gate[-1].bias, math.log(structure['gate_init'] / (1 - structure['gate_init'])))
        nn.init.zeros_(self.geometry_instance_score.weight)
        nn.init.zeros_(self.geometry_instance_score.bias)

    def structural_chain_readout(self, h, mask, graph, papr, rapr):
        """Read physical chain instances before pooling copies of one sequence."""
        base_chain = self.pool_chain(h, mask, papr, rapr)
        if graph['instance_to_chain'].numel() == 0 or graph['positions'].numel() == 0:
            return base_chain
        original = h[graph['chain_index'], graph['positions']]
        geo = self.geometry_input(original) + self.geometry_node(graph['node'])
        for block in self.geometry_blocks:
            geo = block(geo, graph['edge_index'], graph['edge'])
        delta = self.geometry_output(geo)
        node_h = original + self.geometry_node_gate(torch.cat((original, delta), -1)).sigmoid() * delta
        scores = self.geometry_region_score(node_h).squeeze(-1)
        count = len(graph['instance_to_chain'])
        weights = torch.zeros_like(scores)
        for instance in range(count):
            selected = graph['instance_index'] == instance
            if selected.any():
                weights[selected] = scores[selected].softmax(dim=0)
        source, target = graph['edge_index']
        degree = torch.bincount(source, minlength=len(weights)).to(weights.dtype).clamp_min(1)
        neighborhood = torch.zeros_like(weights).index_add(0, target, weights[source] / degree[source])
        spatial_weights = weights + 0.5 * neighborhood
        instance_h = []
        for instance, chain_index in enumerate(graph['instance_to_chain'].tolist()):
            selected = graph['instance_index'] == instance
            if not selected.any():
                instance_h.append(base_chain[chain_index])
                continue
            nodes = node_h[selected]
            regional_weights = spatial_weights[selected]
            regional_weights = regional_weights / regional_weights.sum().clamp_min(1e-08)
            regional = (regional_weights[:, None] * nodes).sum(0)
            summary = torch.cat((nodes.mean(0), regional))
            instance_h.append(base_chain[chain_index] + self.geometry_readout_gate.sigmoid() * self.geometry_readout(summary))
        instance_h = torch.stack(instance_h)
        pooled = []
        for chain_index in range(len(h)):
            selected = instance_h[graph['instance_to_chain'] == chain_index]
            if not len(selected):
                pooled.append(base_chain[chain_index])
            else:
                attention = self.geometry_instance_score(selected).squeeze(-1).softmax(dim=0)
                pooled.append((attention[:, None] * selected).sum(0))
        return torch.stack(pooled)

    def forward(self, esm, mask, chain_to_sample, n_samples, hapr=None, papr=None, rapr=None, graph=None):
        """Cross-Attention -> residue context -> structure -> protein logit."""
        if graph is None:
            raise ValueError('Aligned structure graph is required')
        h = self.esm_proj(esm)
        if hapr is None or papr is None or rapr is None:
            raise ValueError('Current V2 requires HAPR, pAPR and rAPR')
        kv = self.hapr_proj(hapr)
        cross = torch.zeros_like(h)
        for i, length in enumerate(mask.sum(1).tolist()):
            q, k = (h[i:i + 1, :length], kv[i:i + 1, :length])
            result, _ = self.cross_attention(q, k, k, attn_mask=self._relative_bias(length, h.device), need_weights=False)
            cross[i, :length] = result[0]
        gate = self.cross_gate(torch.cat((h, kv, cross, papr[:, :, None], rapr[:, :, None]), -1)).sigmoid() * mask[:, :, None]
        h = self.cross_norm(h + gate * cross + 0.1 * gate * self.hapr_direct(kv))
        h = h.masked_fill(~mask[:, :, None], 0.0)
        h = self.residue_context(h, src_key_padding_mask=~mask)
        h = self.context_norm(h).masked_fill(~mask[:, :, None], 0.0)
        chain_h = self.structural_chain_readout(h, mask, graph, papr, rapr)
        return self.classifier(self.pool_sample(chain_h, chain_to_sample, n_samples)).squeeze(-1)
