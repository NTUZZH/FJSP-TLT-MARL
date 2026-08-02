"""COMA-style learned-critic counterfactual baseline (E2 comparison arm).

Counterfactual advantage per Foerster et al. (AAAI 2018), instantiated for
the event-sequential two-class setting:
    A_t = Q(s_t, a_t) - sum_{a'} pi(a'|s_t) Q(s_t, a')
with a CENTRALIZED learned Q over the acting head's action space (machine
pair grid or vehicle task slots), trained by regression on Monte-Carlo
return-to-go (gamma=1, deterministic env, on-policy — logged in
decisions.md). The actor update keeps the same PPO clipping/entropy as all
other arms (fair comparison: only the advantage estimator differs).

This is exactly the "learned critic, extra network, extra training" cost
that BCB avoids — the arm quantifies that trade.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

sys.path.insert(0, '.')
from model.main_model import DualAttentionNetwork
from model.sub_layers import Actor
from transport_marl.model_transport import DANIELTransport
from transport_marl.mappo import eval_per_row


class CentralQ(nn.Module):
    """Same encoder family as DANIELTransport; heads emit per-action Q."""

    def __init__(self, config):
        super().__init__()
        device = torch.device(config.device)
        self.pair_input_dim = 8
        self.veh_pair_dim = 6
        self.d = config.layer_fea_output_dim[-1]
        self.veh_type_id = config.n_mch_types
        self.op_type_embedding = nn.Embedding(config.n_op_types,
                                              config.type_emb_dim).to(device)
        self.mch_type_embedding = nn.Embedding(config.n_mch_types + 1,
                                               config.type_emb_dim).to(device)
        self.veh_proj = nn.Linear(4, config.fea_m_input_dim).to(device)
        self.feature_exact = DualAttentionNetwork(
            config, config.fea_j_input_dim + config.type_emb_dim,
            config.fea_m_input_dim + config.type_emb_dim).to(device)
        self.q_mch = Actor(config.num_mlp_layers_actor,
                           4 * self.d + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        self.q_veh = Actor(config.num_mlp_layers_actor,
                           4 * self.d + self.veh_pair_dim,
                           config.hidden_dim_actor, 1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                dynamic_pair_mask, fea_pairs, op_type, mch_type,
                fea_v, fea_veh_pairs, veh_action_mask, event_veh, task_dest_op):
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
        fea_mv = torch.cat((fea_m, fea_v_in), dim=1)
        MV = M + V
        mask_mv = torch.ones(B, MV, MV, dtype=mch_mask.dtype, device=mch_mask.device)
        mask_mv[:, :M, :M] = mch_mask
        comp_mv = torch.zeros(B, MV, MV, J, dtype=comp_idx.dtype,
                              device=comp_idx.device)
        comp_mv[:, :M, :M, :] = comp_idx
        fea_j_out, fea_mv_out, gj, gmv = self.feature_exact(
            fea_j, op_mask, candidate, fea_mv, mask_mv, comp_mv)
        fea_m_out = fea_mv_out[:, :M]
        fea_v_out = fea_mv_out[:, M:]
        d = fea_j_out.size(-1)

        cidx = candidate.unsqueeze(-1).repeat(1, 1, d).type(torch.int64)
        Fjc = torch.gather(fea_j_out, 1, cidx)
        Fj_ser = Fjc.unsqueeze(2).repeat(1, 1, M, 1).reshape(B, M * J, d)
        Fm_ser = fea_m_out.unsqueeze(1).repeat(1, J, 1, 1).reshape(B, M * J, d)
        Gj = gj.unsqueeze(1).expand_as(Fj_ser)
        Gm = gmv.unsqueeze(1).expand_as(Fj_ser)
        pair_feat = torch.cat((Fj_ser, Fm_ser, Gj, Gm,
                               fea_pairs.reshape(B, -1, self.pair_input_dim)), dim=-1)
        q_mch = self.q_mch(pair_feat).squeeze(-1)              # [B, J*M]

        dest_idx = task_dest_op.clamp(min=0).unsqueeze(-1).repeat(1, 1, d)
        Fdest = torch.gather(fea_j_out, 1, dest_idx)
        av = event_veh.clamp(min=0)
        Factv = torch.gather(
            fea_v_out, 1, av.view(B, 1, 1).repeat(1, 1, d)).repeat(1, J, 1)
        Gjv = gj.unsqueeze(1).expand_as(Fdest)
        Gmv = gmv.unsqueeze(1).expand_as(Fdest)
        veh_feat = torch.cat((Fdest, Factv, Gjv, Gmv, fea_veh_pairs), dim=-1)
        q_veh = self.q_veh(veh_feat).squeeze(-1)               # [B, J]
        return q_mch, q_veh


def _mc_returns(memory):
    """Monte-Carlo return-to-go per env (gamma=1), masked. [T, E]."""
    r = torch.stack(memory.reward_seq, 0)
    valid = torch.stack(memory.valid_seq, 0).float()
    r = r * valid
    return torch.flip(torch.cumsum(torch.flip(r, [0]), 0), [0])


class COMAPPO:
    """Actor = DANIELTransport (own optimizer); critic = CentralQ.
    Only the advantage estimator differs from TransportPPO."""

    def __init__(self, config):
        self.lr = config.lr
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.tau = config.tau
        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.minibatch_size = config.minibatch_size
        self.policy = DANIELTransport(config)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.critic_q = CentralQ(config)
        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.opt_critic = torch.optim.Adam(self.critic_q.parameters(), lr=self.lr)
        self.device = torch.device(config.device)

    def _forward(self, net, d, sl):
        return net(fea_j=d['fea_j'][sl], op_mask=d['op_mask'][sl],
                   candidate=d['candidate'][sl], fea_m=d['fea_m'][sl],
                   mch_mask=d['mch_mask'][sl], comp_idx=d['comp_idx'][sl],
                   dynamic_pair_mask=d['dynamic_pair_mask'][sl],
                   fea_pairs=d['fea_pairs'][sl], op_type=d['op_type'][sl],
                   mch_type=d['mch_type'][sl], fea_v=d['fea_v'][sl],
                   fea_veh_pairs=d['fea_veh_pairs'][sl],
                   veh_action_mask=d['veh_action_mask'][sl],
                   event_veh=d['event_veh'][sl],
                   task_dest_op=d['task_dest_op'][sl])

    def update(self, memory, credit=None):
        d = memory.flat()
        G = _mc_returns(memory).transpose(0, 1).flatten(0, 1)  # [E*T]
        valid = d['valid'].float()
        B = len(G)
        nb = int(np.ceil(B / self.minibatch_size))
        loss_e, qloss_e = 0.0, 0.0

        for _ in range(self.k_epochs):
            for i in range(nb):
                sl = slice(i * self.minibatch_size,
                           min((i + 1) * self.minibatch_size, B))
                w = valid[sl]
                nvalid = w.sum().clamp(min=1)
                ev = d['event_type'][sl]
                act = d['action'][sl]

                # ---- critic: regress Q(s, a_t) on MC return ----
                q_m, q_v = self._forward(self.critic_q, d, sl)
                q_all = torch.where(
                    ev.unsqueeze(1).bool(),
                    nn.functional.pad(q_v, (0, q_m.size(1) - q_v.size(1))), q_m)
                q_taken = q_all.gather(1, act.unsqueeze(1)).squeeze(1)
                q_loss = (((q_taken - G[sl]) ** 2) * w).sum() / nvalid
                self.opt_critic.zero_grad()
                q_loss.backward()
                self.opt_critic.step()

                # ---- actor: PPO-clipped with COMA advantage ----
                pi_m, pi_v, _ = self._forward(self.policy, d, sl)
                logp, ent = eval_per_row(pi_m, pi_v, ev, act)
                with torch.no_grad():
                    q_m2, q_v2 = self._forward(self.critic_q, d, sl)
                    q_all2 = torch.where(
                        ev.unsqueeze(1).bool(),
                        nn.functional.pad(q_v2, (0, q_m2.size(1) - q_v2.size(1))),
                        q_m2)
                pi_all = torch.where(
                    ev.unsqueeze(1).bool(),
                    nn.functional.pad(pi_v, (0, pi_m.size(1) - pi_v.size(1))),
                    pi_m).detach()
                baseline = (pi_all * q_all2).sum(dim=1)
                q_taken2 = q_all2.gather(1, act.unsqueeze(1)).squeeze(1)
                adv = q_taken2 - baseline
                ratios = torch.exp(logp - d['logprob'][sl].detach())
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                    1 + self.eps_clip) * adv
                p_loss = (-torch.min(surr1, surr2) * w).sum() / nvalid
                e_loss = (-ent * w).sum() / nvalid
                loss = self.ploss_coef * p_loss + self.entloss_coef * e_loss
                self.opt_actor.zero_grad()
                loss.backward()
                self.opt_actor.step()
                loss_e += float(loss.detach())
                qloss_e += float(q_loss.detach())

        for po, p in zip(self.policy_old.parameters(), self.policy.parameters()):
            po.data.copy_(self.tau * po.data + (1 - self.tau) * p.data)
        return loss_e / self.k_epochs, qloss_e / self.k_epochs
