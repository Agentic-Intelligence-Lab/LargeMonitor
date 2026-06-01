import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import TypeVar, Callable, Optional
import logging

import logging
import time
import datetime
import gc

from methods._trainer import _Trainer
from utils.train_utils import select_optimizer, select_scheduler
from utils.memory import MemoryBatchSampler
from torch.utils.data import DataLoader

logger = logging.getLogger()
T = TypeVar('T', bound = 'nn.Module')

class MVP(_Trainer):
    def __init__(self, **kwargs):
        super(MVP, self).__init__(**kwargs)
        
        self.use_mask    = kwargs.get("use_mask")
        self.use_contrastiv  = kwargs.get("use_contrastiv")
        self.use_last_layer  = kwargs.get("use_last_layer")
        self.use_afs  = kwargs.get("use_afs")
        self.use_gsf  = kwargs.get("use_gsf")
        
        self.alpha  = kwargs.get("alpha")
        self.gamma  = kwargs.get("gamma")
        self.margin  = kwargs.get("margin")

        self.labels = torch.empty(0)

        # P0: decaying memory replay boost after task shift (beta=0 disables)
        self.shift_replay_beta = float(kwargs.get("shift_replay_beta", 0.0))
        self.shift_replay_tau = float(kwargs.get("shift_replay_tau", 500.0))
        self.task_shift_detector: Optional[Callable] = kwargs.get("task_shift_detector")
        self.steps_since_shift = 10**9

    def notify_task_shift(self) -> None:
        """Reset replay boost (also used by optional task_shift_detector)."""
        if self.shift_replay_beta <= 0:
            return
        self.steps_since_shift = 0

    def _shift_replay_scale(self) -> float:
        if self.shift_replay_beta <= 0:
            return 1.0
        return 1.0 + self.shift_replay_beta * math.exp(
            -self.steps_since_shift / max(self.shift_replay_tau, 1.0)
        )

    def _effective_memory_batchsize(self) -> int:
        if self.memory_batchsize <= 0:
            return 0
        scaled = int(round(self.memory_batchsize * self._shift_replay_scale()))
        return max(1, min(scaled, self.batchsize - 1))

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        if self.task_shift_detector is not None and self.task_shift_detector(self, images, labels, idx):
            self.notify_task_shift()

        _loss, _acc, _iter = 0.0, 0.0, 0
        memory_bs = self._effective_memory_batchsize()
        memory_iterations = int(self.temp_batchsize * self.online_iter * self.world_size)

        self.memory_sampler = MemoryBatchSampler(self.memory, memory_bs, memory_iterations)
        self.memory_dataloader = DataLoader(
            self.train_dataset, batch_size=memory_bs, sampler=self.memory_sampler, num_workers=4
        )
        self.memory_provider = iter(self.memory_dataloader)

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()], memory_bs)
            _loss += loss
            _acc += acc
            _iter += 1
        self.update_memory(idx, labels)
        self.steps_since_shift += 1
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

    def online_train(self, data, memory_batchsize=None):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        if memory_batchsize is None:
            memory_batchsize = self.memory_batchsize

        x, y = data
        self.labels = torch.cat((self.labels, y), 0)

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        if len(self.memory) > 0 and memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            for i in range(len(memory_labels)):
                memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)
        
        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x, y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            feature, mask = self.model_without_ddp.forward_features(x)
            logit = self.model_without_ddp.forward_head(feature)
            if self.use_mask:
                logit = logit * mask
            logit = logit + self.mask
            loss = self.loss_fn(feature, mask, y)
        return logit, loss

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x)
                logit = logit + self.mask
                loss = F.cross_entropy(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.mean().item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()
            
    def online_before_task(self, task_id):
        # Testing: beta>0 treats each new task as a detected shift (wire real detector later)
        if self.shift_replay_beta > 0:
            self.notify_task_shift()
            print(
                f"[shift-replay] task {task_id} | scale {self._shift_replay_scale():.3f} | "
                f"memory_bs {self._effective_memory_batchsize()} (base {self.memory_batchsize})"
            )

    def online_after_task(self, cur_iter):
        pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.model_without_ddp.backbone.head)
        head.zero_grad()
        logit = head(feature.detach())
        if self.use_mask:
            logit = logit * mask.clone().detach()
        logit = logit + self.mask
        
        sample_loss = F.cross_entropy(logit, y, reduction='none')
        sample_grad = []
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = head.weight.grad[y[idx]].clone()
            sample_grad.append(_g)
            head.zero_grad()
        sample_grad = torch.stack(sample_grad)    #B,dim
        
        head.zero_grad()
        batch_loss = F.cross_entropy(logit, y, reduction='mean')
        batch_loss.backward(retain_graph=True)
        total_batch_grad = head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_grad = total_batch_grad[y[idx]]    #B,dim
        
        return sample_grad, batch_grad
    
    def _get_ignore(self, sample_grad, batch_grad):
        ign_score = (1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1))#B
        return ign_score

    def _get_compensation(self, y, feat):
        head_w = self.model_without_ddp.backbone.head.weight[y].clone().detach()
        cps_score = (1. - torch.cosine_similarity(head_w, feat, dim=1) + self.margin)#B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, feat)
        return ign_score, cps_score
    
    def loss_fn(self, feature, mask, y):
        ign_score, cps_score = self._get_score(feature.detach(), y, mask)

        if self.use_afs:
            logit = self.model_without_ddp.forward_head(feature)
            logit = self.model_without_ddp.forward_head(feature / (cps_score.unsqueeze(1)))
        else:
            logit = self.model_without_ddp.forward_head(feature)
        if self.use_mask:
            logit = logit * mask
        logit = logit + self.mask
        log_p = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(log_p, y)
        if self.use_gsf:
            loss = (1-self.alpha)* loss + self.alpha * (ign_score ** self.gamma) * loss
        return loss.mean() + self.model_without_ddp.get_similarity_loss()
    
    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))} | "
            f"N_Prompts {self.model_without_ddp.e_prompts.size(0)} | "
            f"N_Exposed {len(self.exposed_classes)} | "
            f"Counts {self.model_without_ddp.count.to(torch.int64).tolist()}"
        )

    def setup_distributed_model(self):
        super().setup_distributed_model()
        self.model_without_ddp.use_mask = self.use_mask
        self.model_without_ddp.use_contrastiv = self.use_contrastiv
        self.model_without_ddp.use_last_layer = self.use_last_layer

    def update_memory(self, sample, label):
        # Update memory
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1,)).item()
                    if j < self.memory_size:
                        idx.append(j)
                    else:
                        idx.append(self.memory_size)
        # Distribute idx to all processes
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier() # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        for i, index in enumerate(idx):
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], label[i].item()], index)
            else:
                self.memory.replace_data([sample[i], label[i].item()])