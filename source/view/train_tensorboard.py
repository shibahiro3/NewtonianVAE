from torch.utils.tensorboard import SummaryWriter

import newtonianvae.train
from tool.visualhandlerbase import VisualHandlerBase


class VisualHandler(VisualHandlerBase):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

        self.step = 0

    def plot(self, L, E_ll, E_kl, epoch):
        self.writer.add_scalar("Loss", L, self.step)
        self.writer.add_scalar("NLL", E_ll, self.step)
        self.writer.add_scalar("KL", E_kl, self.step)
        self.writer.flush()  # flush* is not working...
        self.step += 1


if __name__ == "__main__":
    newtonianvae.train.train(VisualHandler("log"))
