import torch
import torch.distributed as dist

class Recorder:
    def __init__(self):
        self.sum = 0
        self.num = 0

    def update(self, item, num=1):
        if item is not None:
            self.sum += item * num
            self.num += num

    def average(self):
        if self.num == 0:
            return None
        return self.sum / self.num

    def clear(self):
        self.sum = 0
        self.num = 0

    def sync(self, device):
        if not dist.is_initialized():
            return

        sum_tensor = torch.tensor(float(self.sum), device=device)
        num_tensor = torch.tensor(float(self.num), device=device)

        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_tensor, op=dist.ReduceOp.SUM)

        self.sum = sum_tensor.item()
        self.num = num_tensor.item()