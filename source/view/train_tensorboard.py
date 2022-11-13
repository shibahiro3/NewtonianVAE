from torch.utils.tensorboard import SummaryWriter

import nvae.train
from tool.visualhandlerbase import VisualHandlerBase


class VisualHandler(VisualHandlerBase):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

        self.step = 0

    def plot(self, L, LOG_E_ll_sum, LOG_E_kl_sum, epoch):
        self.writer.add_scalar("Loss", L, self.step)
        self.writer.add_scalar("NLL", LOG_E_ll_sum, self.step)
        self.writer.add_scalar("KL", LOG_E_kl_sum, self.step)
        self.writer.flush()  # flush* is not working...
        self.step += 1


if __name__ == "__main__":
    nvae.train.train(VisualHandler("log"))
