import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Optional, Sized, Iterable, Tuple

class OnlineSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], num_tasks: int, m: int, n: int,  rnd_seed: int, cur_iter: int= 0, varing_NM: bool= False, num_replicas: int=None, rank: int=None) -> None:

        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        
        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:
                self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]
            # self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            # self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            if num_blurred > 0:
                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                # num_blurred = len(blurred) // num_tasks
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int)-> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))
    
    def get_task(self, cur_iter: int)-> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

class OnlineBatchSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], num_tasks: int, m: int, n: int, rnd_seed: int, batchsize: int=16, online_iter: int=1, cur_iter: int=0, varing_NM: bool=False, num_replicas: int=None, rank: int=None) -> None:
        super().__init__(data_source)
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.num_tasks  = num_tasks
        self.m      = m
        self.n      = n
        self.rnd_seed   = rnd_seed
        self.batchsize  = batchsize
        self.online_iter    = online_iter
        self.cur_iter   = cur_iter
        self.varing_NM  = varing_NM

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)]
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            # num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:

        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))
    
    def get_task(self, cur_iter : int) -> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

    def get_task_classes(self, cur_iter : int) -> Iterable[int]:
        return list(set(self.classes[self.indices[cur_iter]]))


class ImageNetHSDisjointSampler(Sampler):
    """Sample by HS stream step (T1–T10), not Si-Blurry class splits.

    Requires ``stream_task_ids`` on the underlying dataset (``ImageNet_HS``).
    ``num_tasks=10``: one training session per stream step.
    ``num_tasks=5``: merge each group's two consecutive steps (tiny + shift).
    """

    HS_STREAM_STEPS = 10

    def __init__(
        self,
        data_source: Optional[Sized],
        num_tasks: int = 10,
        rnd_seed: int = 0,
        cur_iter: int = 0,
        num_replicas: int = None,
        rank: int = None,
    ) -> None:
        from imagenet_hs import TASK_SPECS

        self.data_source = data_source
        self.classes = self.data_source.classes
        self.targets = self.data_source.targets
        self.task_specs = TASK_SPECS
        self.generator = torch.Generator().manual_seed(rnd_seed)
        self.num_tasks = num_tasks
        self.task = cur_iter

        base = getattr(data_source, "dataset", data_source)
        stream_task_ids = getattr(base, "stream_task_ids", None)
        if stream_task_ids is None:
            raise AttributeError(
                "ImageNetHSDisjointSampler requires stream_task_ids on the dataset "
                "(use ImageNet_HS)."
            )

        if num_tasks not in (5, 10):
            raise ValueError(
                f"imagenet-hs supports num_tasks 5 (merged groups) or 10 (native stream), "
                f"got {num_tasks}"
            )

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        by_stream = [[] for _ in range(self.HS_STREAM_STEPS)]
        for i, tid in enumerate(stream_task_ids):
            by_stream[tid].append(i)

        if num_tasks == 10:
            self.stream_steps_for_task = [[t] for t in range(self.HS_STREAM_STEPS)]
        else:
            self.stream_steps_for_task = [[2 * t, 2 * t + 1] for t in range(5)]

        self.indices = []
        print("ImageNet-HS stream schedule:")
        for task_id, steps in enumerate(self.stream_steps_for_task):
            merged = []
            for step in steps:
                merged.extend(by_stream[step])
            self.indices.append(self._shuffled(merged))
            spec = TASK_SPECS[steps[-1]]
            step_label = "+".join(f"T{s + 1}" for s in steps)
            print(
                f"  train task {task_id} ({step_label}): "
                f"group={spec.group} source={spec.source} "
                f"change={spec.change_type} n={len(self.indices[-1])}"
            )

        self._refresh_task_sizes()

    def _shuffled(self, indices: list[int]) -> list[int]:
        if not indices:
            return []
        idx = torch.tensor(indices)
        return idx[torch.randperm(len(idx), generator=self.generator)].tolist()

    def _refresh_task_sizes(self) -> None:
        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def change_type_for_task(self, task_id: int) -> str:
        steps = self.stream_steps_for_task[task_id]
        return self.task_specs[steps[-1]].change_type

    def stream_spec_for_task(self, task_id: int):
        steps = self.stream_steps_for_task[task_id]
        return self.task_specs[steps[-1]]

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter
        self._refresh_task_sizes()

    def get_task(self, cur_iter: int) -> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]


class OnlineTestSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], exposed_class : Iterable[int], num_replicas: int=None, rank: int=None) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        if self.distributed:
            self.num_samples = int(len(self.indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self) -> int:
        return self.num_selected_samples