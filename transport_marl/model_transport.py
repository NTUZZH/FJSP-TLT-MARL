"""DANIEL-Transport: the lab dual-attention backbone with a VEHICLE node set
in the machine-side attention stream + a second (vehicle-class) actor head
(Paper X2, proposal §6).

- Vehicle raw features [E, V, 4] -> linear proj to the machine feature width,
  + type embedding: the mch-type vocabulary is extended by one VEHICLE id
  (index n_mch_types); machines/vehicles share one attention stream, so
  n_mch_types passed to the embedding is n_mch_types + 1.
- comp_idx (machine-competition edges) is zero-padded on vehicle rows/cols;
  mch_mask extended so vehicles attend to and are attended by all live nodes.
- Machine head: the house pair head, unchanged inputs [op, mch, Gj, Gmv, pair].
- Vehicle head: per-job task slots scored from [dest-op emb, acting-vehicle
  emb, Gj, Gmv, fea_veh_pairs(6)].
- ONE centralized critic on [Gj, Gmv] (CTDE / MAPPO).

Both heads are computed every forward; rows where a head is irrelevant have
fully-masked grids -- those rows get action 0 force-unmasked to keep softmax
finite, and the trainer routes losses by event_type so they never contribute.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from common_utils import nonzero_averaging
from model.main_model import DualAttentionNetwork
from model.sub_layers import Actor, Critic


def _safe_mask_softmax(scores, mask):
    """softmax with -inf masking; fully-masked rows fall back to action 0."""
    all_masked = mask.all(dim=1)
    mask = mask.clone()
    mask[all_masked, 0] = False
    scores = scores.masked_fill(mask, float('-inf'))
    return F.softmax(scores, dim=1)


class DANIELTransport(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = torch.device(config.device)
        # +1 guide channel on both pair grids when the bound-guided action
        # prior is enabled (transport_marl/guide.py; snapshot key 'guide')
        guide = 1 if getattr(config, 'guide', False) else 0
        self.pair_input_dim = 8 + guide
        self.veh_pair_dim = 6 + guide
        self.d = config.layer_fea_output_dim[-1]

        self.type_emb_dim = config.type_emb_dim
        self.op_type_embedding = nn.Embedding(config.n_op_types,
                                              config.type_emb_dim).to(device)
        # +1 type id for vehicles (shared machine-stream embedding)
        self.veh_type_id = config.n_mch_types
        self.mch_type_embedding = nn.Embedding(config.n_mch_types + 1,
                                               config.type_emb_dim).to(device)

        fea_j_input_dim = config.fea_j_input_dim + config.type_emb_dim
        fea_m_input_dim = config.fea_m_input_dim + config.type_emb_dim
        self.veh_proj = nn.Linear(4, config.fea_m_input_dim).to(device)

        self.feature_exact = DualAttentionNetwork(
            config, fea_j_input_dim, fea_m_input_dim).to(device)
        self.actor = Actor(config.num_mlp_layers_actor,
                           4 * self.d + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        self.actor_veh = Actor(config.num_mlp_layers_actor,
                               4 * self.d + self.veh_pair_dim,
                               config.hidden_dim_actor, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.d,
                             config.hidden_dim_critic, 1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                dynamic_pair_mask, fea_pairs, op_type, mch_type,
                fea_v, fea_veh_pairs, veh_action_mask, event_veh, task_dest_op):
        """
        Shapes: fea_j [B,N,fj], fea_m [B,M,fm], fea_v [B,V,4],
        fea_veh_pairs [B,J,6], veh_action_mask [B,J] (True=masked),
        event_veh [B] (-1 if machine event), task_dest_op [B,J] (-1 inactive).
        Returns pi_mch [B, J*M], pi_veh [B, J], v [B,1].
        """
        B, M, _, J = comp_idx.size()
        V = fea_v.size(1)

        live_j = (fea_j.abs().sum(dim=-1, keepdim=True) > 0).float()
        live_m = (fea_m.abs().sum(dim=-1, keepdim=True) > 0).float()
        fea_j = torch.cat((fea_j, self.op_type_embedding(op_type) * live_j), dim=-1)
        fea_m = torch.cat((fea_m, self.mch_type_embedding(mch_type) * live_m), dim=-1)

        veh_type = torch.full((B, V), self.veh_type_id, dtype=torch.long,
                              device=fea_v.device)
        fea_v_in = torch.cat((self.veh_proj(fea_v),
                              self.mch_type_embedding(veh_type)), dim=-1)

        # join machine stream: nodes [B, M+V, fm+emb]
        fea_mv = torch.cat((fea_m, fea_v_in), dim=1)
        MV = M + V
        mask_mv = torch.ones(B, MV, MV, dtype=mch_mask.dtype, device=mch_mask.device)
        mask_mv[:, :M, :M] = mch_mask
        comp_mv = torch.zeros(B, MV, MV, J, dtype=comp_idx.dtype,
                              device=comp_idx.device)
        comp_mv[:, :M, :M, :] = comp_idx

        fea_j_out, fea_mv_out, fea_j_global, fea_mv_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_mv, mask_mv, comp_mv)
        fea_m_out = fea_mv_out[:, :M]
        fea_v_out = fea_mv_out[:, M:]
        d = fea_j_out.size(-1)

        # ----- machine head (house pair grid, order j*M + m) -----
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d).type(torch.int64)
        Fea_j_JC = torch.gather(fea_j_out, 1, candidate_idx)
        Fea_j_ser = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(B, M * J, d)
        Fea_m_ser = fea_m_out.unsqueeze(1).repeat(1, J, 1, 1).reshape(B, M * J, d)
        Gj = fea_j_global.unsqueeze(1).expand_as(Fea_j_ser)
        Gm = fea_mv_global.unsqueeze(1).expand_as(Fea_j_ser)
        pair_feat = torch.cat((Fea_j_ser, Fea_m_ser, Gj, Gm,
                               fea_pairs.reshape(B, -1, self.pair_input_dim)), dim=-1)
        mch_scores = self.actor(pair_feat).squeeze(-1)
        pi_mch = _safe_mask_softmax(mch_scores, dynamic_pair_mask.reshape(B, -1))

        # ----- vehicle head (job task slots) -----
        dest_idx = task_dest_op.clamp(min=0).unsqueeze(-1).repeat(1, 1, d)
        Fea_dest = torch.gather(fea_j_out, 1, dest_idx)              # [B, J, d]
        av = event_veh.clamp(min=0)
        Fea_actv = torch.gather(
            fea_v_out, 1, av.view(B, 1, 1).repeat(1, 1, d)).repeat(1, J, 1)
        Gj_v = fea_j_global.unsqueeze(1).expand_as(Fea_dest)
        Gm_v = fea_mv_global.unsqueeze(1).expand_as(Fea_dest)
        veh_feat = torch.cat((Fea_dest, Fea_actv, Gj_v, Gm_v, fea_veh_pairs), dim=-1)
        veh_scores = self.actor_veh(veh_feat).squeeze(-1)
        pi_veh = _safe_mask_softmax(veh_scores, veh_action_mask)

        v = self.critic(torch.cat((fea_j_global, fea_mv_global), dim=-1))
        return pi_mch, pi_veh, v
