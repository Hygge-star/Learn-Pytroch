from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("log")
#writer.add_image()
#y=2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, global_step=i)

writer.close()
