"""MVP with shift-type-aware policies for ImageNet-HS (oracle mode).

Uses ``ImageNetHSDisjointSampler.change_type_for_task`` to select actions:
  - initial / new_class: stronger replay + full GSF
  - domain_shift: moderate replay + AFS on, GSF damped
  - corruption: light replay + GSF off + conservative memory writes
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from methods.mvp import MVP


@dataclass(frozen=True)
class ShiftAction:
    replay_beta: float
    alpha_scale: float
    use_afs: Optional[bool]
    use_gsf: Optional[bool]
    memory_skip_prob: float


DEFAULT_SHIFT_POLICIES: dict[str, ShiftAction] = {
    "initial": ShiftAction(
        replay_beta=1.0,
        alpha_scale=1.0,
        use_afs=None,
        use_gsf=None,
        memory_skip_prob=0.0,
    ),
    "new_class": ShiftAction(
        replay_beta=1.0,
        alpha_scale=1.0,
        use_afs=None,
        use_gsf=True,
        memory_skip_prob=0.0,
    ),
    "domain_shift": ShiftAction(
        replay_beta=0.5,
        alpha_scale=0.3,
        use_afs=True,
        use_gsf=True,
        memory_skip_prob=0.0,
    ),
    "corruption": ShiftAction(
        replay_beta=0.3,
        alpha_scale=0.1,
        use_afs=False,
        use_gsf=False,
        memory_skip_prob=0.5,
    ),
}


class MVPShift(MVP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shift_policy_mode = kwargs.get("shift_policy_mode", "oracle")
        self.current_change_type = "new_class"
        self._active_replay_beta = 0.0
        self._policy_alpha_scale = 1.0
        self._policy_use_afs: Optional[bool] = None
        self._policy_use_gsf: Optional[bool] = None
        self._memory_skip_prob = 0.0

    def _resolve_change_type(self, task_id: int) -> str:
        if self.shift_policy_mode != "oracle":
            return "new_class"
        sampler = getattr(self, "train_sampler", None)
        if sampler is not None and hasattr(sampler, "change_type_for_task"):
            return sampler.change_type_for_task(task_id)
        return "new_class"

    def _apply_shift_policy(self, change_type: str) -> None:
        policy = DEFAULT_SHIFT_POLICIES.get(change_type, DEFAULT_SHIFT_POLICIES["new_class"])
        self.current_change_type = change_type
        self._active_replay_beta = policy.replay_beta
        self._policy_alpha_scale = policy.alpha_scale
        self._policy_use_afs = policy.use_afs
        self._policy_use_gsf = policy.use_gsf
        self._memory_skip_prob = policy.memory_skip_prob

    def _effective_use_afs(self) -> bool:
        if self._policy_use_afs is not None:
            return self._policy_use_afs
        return bool(self.use_afs)

    def _effective_use_gsf(self) -> bool:
        if self._policy_use_gsf is not None:
            return self._policy_use_gsf
        return bool(self.use_gsf)

    def _effective_alpha(self) -> float:
        return float(self.alpha) * self._policy_alpha_scale

    def _shift_iter_scale(self) -> float:
        if self._active_replay_beta == 0:
            return 1.0
        return 1.0 + self._active_replay_beta * math.exp(
            -self.steps_since_shift / max(self.shift_replay_tau, 1.0)
        )

    def notify_task_shift(self) -> None:
        if self._active_replay_beta == 0:
            return
        self.steps_since_shift = 0

    def online_before_task(self, task_id: int) -> None:
        change_type = self._resolve_change_type(task_id)
        self._apply_shift_policy(change_type)
        if self._active_replay_beta != 0:
            self.notify_task_shift()
        spec = None
        sampler = getattr(self, "train_sampler", None)
        if sampler is not None and hasattr(sampler, "stream_spec_for_task"):
            spec = sampler.stream_spec_for_task(task_id)
        print(
            f"[mvp-shift] task {task_id} | change={change_type} | "
            f"source={getattr(spec, 'source', '?')} group={getattr(spec, 'group', '?')} | "
            f"replay_beta={self._active_replay_beta} | alpha={self._effective_alpha():.3f} | "
            f"afs={self._effective_use_afs()} gsf={self._effective_use_gsf()} | "
            f"mem_skip={self._memory_skip_prob:.2f} | "
            f"online_iter {self._effective_online_iter()} (base {int(self.online_iter)})"
        )

    def loss_fn(self, feature, mask, y):
        ign_score, cps_score = self._get_score(feature.detach(), y, mask)
        use_afs = self._effective_use_afs()
        use_gsf = self._effective_use_gsf()
        alpha = self._effective_alpha()

        if use_afs:
            logit = self.model_without_ddp.forward_head(feature)
            logit = self.model_without_ddp.forward_head(feature / (cps_score.unsqueeze(1)))
        else:
            logit = self.model_without_ddp.forward_head(feature)
        if self.use_mask:
            logit = logit * mask
        logit = logit + self.mask
        log_p = torch.nn.functional.log_softmax(logit, dim=1)
        loss = torch.nn.functional.nll_loss(log_p, y)
        if use_gsf:
            loss = (1 - alpha) * loss + alpha * (ign_score ** self.gamma) * loss
        return loss.mean() + self.model_without_ddp.get_similarity_loss()

    def update_memory(self, sample, label):
        if self._memory_skip_prob <= 0:
            return super().update_memory(sample, label)

        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()

        idx = []
        if self.is_main_process():
            for lbl in label:
                if random.random() < self._memory_skip_prob:
                    idx.append(self.memory_size)
                    continue
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1,)).item()
                    if j < self.memory_size:
                        idx.append(j)
                    else:
                        idx.append(self.memory_size)

        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier()
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()

        for i, index in enumerate(idx):
            if index >= self.memory_size:
                continue
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], label[i].item()], index)
            else:
                self.memory.replace_data([sample[i], label[i].item()])
