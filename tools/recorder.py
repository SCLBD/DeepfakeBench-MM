import torch
import torch.distributed as dist


class Recorder:
    """Accumulate weighted scalar values with optional distributed reduction."""

    def __init__(self):
        self.sum = 0
        self.num = 0

    def update(self, item, num=1):
        """Accumulate item * num and its weight, skipping None items.

        Args:
            item: Scalar value to record, or None to skip the update.
            num: Weight associated with the value, usually the batch size.
        """
        if item is not None:
            self.sum += item * num
            self.num += num

    def average(self):
        """Return the weighted average, or None when no weight has been recorded."""
        if self.num == 0:
            return None
        return self.sum / self.num

    def clear(self):
        """Reset the accumulated sum and weight."""
        self.sum = 0
        self.num = 0

    def sync(self, device):
        """Sum accumulated values and weights across an initialized process group.

        Args:
            device: Device used to allocate the reduction tensors.
        """
        if not dist.is_initialized():
            return

        sum_tensor = torch.tensor(float(self.sum), device=device)
        num_tensor = torch.tensor(float(self.num), device=device)

        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_tensor, op=dist.ReduceOp.SUM)

        self.sum = sum_tensor.item()
        self.num = num_tensor.item()
