"""Single-agent joint baseline for FJSP-TL-T (Paper X2, §8 E4).

The fair single-agent contrast to the MARL model (DANIELTransport). Held FIXED
vs the MARL model: the shared DANIEL dual-attention encoder over op/machine/
vehicle nodes, the ONE centralized critic on the pooled global, the Theorem-1
telescoped bound reward, and every PPO hyperparameter/optimizer detail.

The ONLY change is the actor: the MARL model's TWO parameter-shared,
class-specialized heads (a machine-pair head of input width 4d+8 and a
vehicle-task head of input width 4d+6) are replaced by ONE unified actor MLP.
Because the two candidate-action layouts have DIFFERENT raw widths (40 vs 38
at d=8), each is first mapped by a thin learned linear adapter into a common
candidate-feature dimension `cand_dim`; the SAME Actor MLP then scores both.
At a machine event it scores the J*M pairs; at a vehicle event it scores the J
task slots; masking is exactly what the env dictates per event (reusing the
MARL model's `_safe_mask_softmax` so the masking is bit-identical).

The forward signature and return (pi_mch [B,J*M], pi_veh [B,J], v [B,1]) match
DANIELTransport, so `select_actions` / `eval_per_row` route by event_type
unchanged and TransportPPO's masked-GAE update is reused verbatim (SinglePPO
below only swaps the policy network). This removes the per-class head
specialization while holding encoder/critic/reward/optimizer fixed, isolating
the effect of the agent decomposition -- the E4 question.
"""

import sys
import torch
import torch.nn as nn
from copy import deepcopy

sys.path.insert(0, '.')
from model.main_model import DualAttentionNetwork
from model.sub_layers import Actor, Critic
from transport_marl.model_transport import _safe_mask_softmax
from transport_marl.mappo import TransportPPO


class DANIELSingle(nn.Module):
    """Unified-head single-agent policy: one Actor MLP scores both action
    classes over a common candidate-feature space. Encoder + critic identical
    to DANIELTransport."""

    def __init__(self, config):
        super().__init__()
        device = torch.device(config.device)
        # the guide channel widens both pair grids by one, exactly as in
        # DANIELTransport; without this the merged head cannot be trained
        # with the certified action prices, which would leave the
        # single-agent control differing from the proposed arm in two ways
        # at once, the head structure and the price channel
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

        # raw candidate widths of the two action classes (as in DANIELTransport)
        mch_cand_in = 4 * self.d + self.pair_input_dim   # 40 at d=8
        veh_cand_in = 4 * self.d + self.veh_pair_dim     # 38 at d=8
        # common candidate-feature dimension (== the richer machine-pair width);
        # thin learned adapters project both classes into it so ONE Actor MLP
        # (the real policy head) scores both -- the single-agent monolith.
        self.cand_dim = mch_cand_in
        self.mch_cand_proj = nn.Linear(mch_cand_in, self.cand_dim).to(device)
        self.veh_cand_proj = nn.Linear(veh_cand_in, self.cand_dim).to(device)
        self.actor = Actor(config.num_mlp_layers_actor, self.cand_dim,
                           config.hidden_dim_actor, 1).to(device)

        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.d,
                             config.hidden_dim_critic, 1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                dynamic_pair_mask, fea_pairs, op_type, mch_type,
                fea_v, fea_veh_pairs, veh_action_mask, event_veh, task_dest_op):
        """Same shapes as DANIELTransport.forward. Returns pi_mch [B, J*M],
        pi_veh [B, J], v [B, 1]."""
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

        # ----- machine candidate features (house pair grid, order j*M + m) -----
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d).type(torch.int64)
        Fea_j_JC = torch.gather(fea_j_out, 1, candidate_idx)
        Fea_j_ser = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(B, M * J, d)
        Fea_m_ser = fea_m_out.unsqueeze(1).repeat(1, J, 1, 1).reshape(B, M * J, d)
        Gj = fea_j_global.unsqueeze(1).expand_as(Fea_j_ser)
        Gm = fea_mv_global.unsqueeze(1).expand_as(Fea_j_ser)
        pair_feat = torch.cat((Fea_j_ser, Fea_m_ser, Gj, Gm,
                               fea_pairs.reshape(B, -1, self.pair_input_dim)), dim=-1)

        # ----- vehicle candidate features (job task slots) -----
        dest_idx = task_dest_op.clamp(min=0).unsqueeze(-1).repeat(1, 1, d)
        Fea_dest = torch.gather(fea_j_out, 1, dest_idx)              # [B, J, d]
        av = event_veh.clamp(min=0)
        Fea_actv = torch.gather(
            fea_v_out, 1, av.view(B, 1, 1).repeat(1, 1, d)).repeat(1, J, 1)
        Gj_v = fea_j_global.unsqueeze(1).expand_as(Fea_dest)
        Gm_v = fea_mv_global.unsqueeze(1).expand_as(Fea_dest)
        veh_feat = torch.cat((Fea_dest, Fea_actv, Gj_v, Gm_v, fea_veh_pairs), dim=-1)

        # ----- ONE unified head over a common candidate-feature space -----
        mch_cand = self.mch_cand_proj(pair_feat)          # [B, J*M, cand_dim]
        veh_cand = self.veh_cand_proj(veh_feat)           # [B, J,   cand_dim]
        mch_scores = self.actor(mch_cand).squeeze(-1)
        veh_scores = self.actor(veh_cand).squeeze(-1)
        pi_mch = _safe_mask_softmax(mch_scores, dynamic_pair_mask.reshape(B, -1))
        pi_veh = _safe_mask_softmax(veh_scores, veh_action_mask)

        v = self.critic(torch.cat((fea_j_global, fea_mv_global), dim=-1))
        return pi_mch, pi_veh, v


class SinglePPO(TransportPPO):
    """TransportPPO with the unified-head DANIELSingle policy swapped in.

    Only the policy network changes. Because DANIELSingle keeps the
    (pi_mch, pi_veh, v) signature, TransportPPO's `_forward`, `select_actions`
    / `eval_per_row` event routing, masked-GAE update, PPO clipping, entropy,
    optimizer, and polyak target update are all inherited verbatim -- exactly
    the fairness the E4 contrast requires (same critic/reward/optimizer, only
    the agent decomposition differs)."""

    def __init__(self, config):
        super().__init__(config)          # builds the shared PPO machinery
        self.policy = DANIELSingle(config)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
