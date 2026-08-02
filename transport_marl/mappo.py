"""MAPPO trainer for FJSP-TL-T (Paper X2): two parameter-shared actor heads
(machine-class, vehicle-class) on one shared encoder + one centralized
critic, CTDE. Adapted from model/PPO.py (house hyperparameters unchanged,
Appendix B); differences:

- stores the transport state tensors and event routing info per step;
- episodes have DIFFERENT lengths across envs -> a `valid` mask marks real
  decisions (post-done pushes are padding); GAE, advantage normalization,
  and all losses are computed under that mask;
- per-row policy terms come from the head selected by event_type.

Shared team reward (P2 default: Theorem-1 telescoped bound difference).
Per-agent credit (BCB, M1) lands in P3 as an additional baseline term.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

sys.path.insert(0, '.')
from transport_marl.model_transport import DANIELTransport


class TransportMemory:
    FIELDS = ['fea_j', 'op_mask', 'fea_m', 'mch_mask', 'dynamic_pair_mask',
              'comp_idx', 'candidate', 'fea_pairs', 'op_type', 'mch_type',
              'fea_v', 'fea_veh_pairs', 'veh_action_mask', 'event_type',
              'event_veh', 'task_dest_op']

    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear_memory()

    def clear_memory(self):
        self.seq = {f: [] for f in self.FIELDS}
        self.action_seq = []
        self.reward_seq = []
        self.val_seq = []
        self.done_seq = []
        self.log_probs = []
        self.valid_seq = []
        self.rpass_seq = []   # M1/BCB: r_t^pass per step (state-event baseline)
        # policy mask: False = forced/rule step (real transition: participates
        # in GAE and the value loss, but not in the policy/entropy loss).
        # Used by the E0b arm (fixed vehicle rule).
        self.polmask_seq = []

    def push(self, state, valid):
        s = self.seq
        s['fea_j'].append(state.fea_j_tensor)
        s['op_mask'].append(state.op_mask_tensor)
        s['fea_m'].append(state.fea_m_tensor)
        s['mch_mask'].append(state.mch_mask_tensor)
        s['dynamic_pair_mask'].append(state.dynamic_pair_mask_tensor)
        s['comp_idx'].append(state.comp_idx_tensor)
        s['candidate'].append(state.candidate_tensor)
        s['fea_pairs'].append(state.fea_pairs_tensor)
        s['op_type'].append(state.op_type_tensor)
        s['mch_type'].append(state.mch_type_tensor)
        s['fea_v'].append(state.fea_v_tensor)
        s['fea_veh_pairs'].append(state.fea_veh_pairs_tensor)
        s['veh_action_mask'].append(state.veh_action_mask_tensor)
        s['event_type'].append(state.event_type_tensor)
        s['event_veh'].append(state.event_veh_tensor)
        s['task_dest_op'].append(state.task_dest_op_tensor)
        self.valid_seq.append(valid)   # torch bool [E]

    def flat(self):
        """[T,E,...] -> flattened [E*T, ...] dict + aux tensors."""
        out = {}
        for f in self.FIELDS:
            out[f] = torch.stack(self.seq[f], dim=0).transpose(0, 1).flatten(0, 1)
        out['action'] = torch.stack(self.action_seq, 0).transpose(0, 1).flatten(0, 1)
        out['logprob'] = torch.stack(self.log_probs, 0).transpose(0, 1).flatten(0, 1)
        out['valid'] = torch.stack(self.valid_seq, 0).transpose(0, 1).flatten(0, 1)
        if self.polmask_seq:
            out['polmask'] = torch.stack(self.polmask_seq, 0).transpose(0, 1).flatten(0, 1)
        else:
            out['polmask'] = torch.ones_like(out['valid'])
        return out

    def gae(self, credit='shared'):
        """Masked GAE. Returns flattened advantages + value targets [E*T].

        credit='m1' (BCB): subtract the analytic pass-counterfactual reward
        r_t^pass from each step's advantage BEFORE normalization. r_t^pass is
        a function of (state, event) only, so the policy gradient stays
        unbiased (Lemma 1); the value target is untouched.
        credit='m2' (ablation arm): REPLACE the team reward with the per-agent
        difference signal r_t - r_t^pass (Wolpert-Tumer instantiation with an
        analytic G). Policy invariance is NOT guaranteed and never claimed;
        the critic learns the shaped return."""
        r = torch.stack(self.reward_seq, 0)          # [T, E]
        v = torch.stack(self.val_seq, 0)             # [T, E]
        d = torch.stack(self.done_seq, 0)            # [T, E] 1.0 = done AFTER step
        valid = torch.stack(self.valid_seq, 0).float()
        T, E = r.shape
        if credit == 'm2':
            assert len(self.rpass_seq) == T, 'M2 needs r_pass at every step'
            r = (r - torch.stack(self.rpass_seq, 0)) * valid
        adv = torch.zeros(E, device=r.device)
        adv_seq = []
        for t in reversed(range(T)):
            nonterm = 1.0 - d[t]
            v_next = v[t + 1] if t + 1 < T else torch.zeros_like(v[0])
            delta = r[t] + self.gamma * v_next * nonterm - v[t]
            adv = delta + self.gamma * self.gae_lambda * nonterm * adv
            adv = adv * valid[t]                     # padding contributes nothing
            adv_seq.insert(0, adv)
        A = torch.stack(adv_seq, 0)                  # [T, E]
        vt = A + v
        if credit == 'm1':
            assert len(self.rpass_seq) == T, 'M1 needs r_pass at every step'
            rp = torch.stack(self.rpass_seq, 0)      # [T, E]
            A = (A - rp) * valid
        # masked per-env normalization (house normalizes per env across time)
        n = valid.sum(0).clamp(min=1)
        mean = (A * valid).sum(0) / n
        var = ((A - mean).pow(2) * valid).sum(0) / n
        A = (A - mean) / (var.sqrt() + 1e-8)
        A = A * valid
        return (A.transpose(0, 1).flatten(0, 1),
                vt.transpose(0, 1).flatten(0, 1))


def select_actions(pi_mch, pi_veh, event_type, greedy=False):
    """Route sampling by event type. Returns (actions [B], logprobs [B])."""
    pi = torch.where(event_type.unsqueeze(1).bool(),
                     nn.functional.pad(pi_veh, (0, pi_mch.size(1) - pi_veh.size(1))),
                     pi_mch)
    if greedy:
        a = torch.argmax(pi, dim=1)
    else:
        a = torch.multinomial(pi, 1).squeeze(1)   # zero-prob rows never sampled
    logp = torch.log(pi.gather(1, a.unsqueeze(1)).squeeze(1).clamp(min=1e-12))
    return a, logp


def eval_per_row(pi_mch, pi_veh, event_type, actions):
    """Per-row logprob + entropy from the routed head."""
    B = pi_mch.size(0)
    pi = torch.where(event_type.unsqueeze(1).bool(),
                     nn.functional.pad(pi_veh, (0, pi_mch.size(1) - pi_veh.size(1))),
                     pi_mch)
    logp = torch.log(pi.gather(1, actions.unsqueeze(1)).squeeze(1).clamp(min=1e-12))
    plogp = pi * torch.log(pi.clamp(min=1e-12))
    ent = -plogp.sum(dim=1)
    return logp, ent


class TransportPPO:
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss(reduction='none')
        self.device = torch.device(config.device)

    def _forward(self, policy, d, sl):
        return policy(fea_j=d['fea_j'][sl], op_mask=d['op_mask'][sl],
                      candidate=d['candidate'][sl], fea_m=d['fea_m'][sl],
                      mch_mask=d['mch_mask'][sl], comp_idx=d['comp_idx'][sl],
                      dynamic_pair_mask=d['dynamic_pair_mask'][sl],
                      fea_pairs=d['fea_pairs'][sl], op_type=d['op_type'][sl],
                      mch_type=d['mch_type'][sl], fea_v=d['fea_v'][sl],
                      fea_veh_pairs=d['fea_veh_pairs'][sl],
                      veh_action_mask=d['veh_action_mask'][sl],
                      event_veh=d['event_veh'][sl],
                      task_dest_op=d['task_dest_op'][sl])

    def update(self, memory, credit='shared'):
        d = memory.flat()
        A, vt = memory.gae(credit=credit)
        valid = d['valid'].float()
        B = len(A)
        nb = int(np.ceil(B / self.minibatch_size))
        loss_epochs, v_loss_epochs = 0.0, 0.0

        for _ in range(self.k_epochs):
            for i in range(nb):
                sl = slice(i * self.minibatch_size,
                           min((i + 1) * self.minibatch_size, B))
                pi_m, pi_v, vals = self._forward(self.policy, d, sl)
                logp, ent = eval_per_row(pi_m, pi_v, d['event_type'][sl],
                                         d['action'][sl])
                w = valid[sl]
                wp = w * d['polmask'][sl].float()
                nvalid = w.sum().clamp(min=1)
                npol = wp.sum().clamp(min=1)
                ratios = torch.exp(logp - d['logprob'][sl].detach())
                adv = A[sl]
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                    1 + self.eps_clip) * adv
                p_loss = (-torch.min(surr1, surr2) * wp).sum() / npol
                v_loss = (self.V_loss_2(vals.squeeze(1), vt[sl]) * w).sum() / nvalid
                e_loss = (-ent * wp).sum() / npol
                loss = (self.vloss_coef * v_loss + self.ploss_coef * p_loss
                        + self.entloss_coef * e_loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epochs += float(loss.detach())
                v_loss_epochs += float(v_loss.detach())

        for po, p in zip(self.policy_old.parameters(), self.policy.parameters()):
            po.data.copy_(self.tau * po.data + (1 - self.tau) * p.data)
        return loss_epochs / self.k_epochs, v_loss_epochs / self.k_epochs
