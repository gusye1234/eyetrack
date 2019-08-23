from load_data import dataloader
from model import ITrackerModel, ItrackerImageModel
import utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import world
from parser import args




if __name__ == "__main__":
    args = args().parse_args()
    world.batch_size = args.batch_size
    world.doLoad = args.doload    
    
    tran = transforms.Compose([utils.Scale() ,utils.ToTensor()])
    data = dataloader(transform=tran)
    data_train = DataLoader(data, batch_size=world.batch_size, shuffle=True, num_workers=world.workers)
    world.n_batch = len(data_train)
    data_test = dataloader(mode="test", transform=tran)
    data_test = DataLoader(data_test, batch_size=world.batch_size, shuffle=True, num_workers=world.workers)

    data_sample = data[0]
    img = utils.make_sample(data_sample)
    utils.show_config()

    m = ITrackerModel()
    epoch_before = 0
    best = 1e10
    if world.doLoad:
        saved = utils.load_checkpoint()
        if saved:
            print('>Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                m.module.load_state_dict(state)
            except:
                m.load_state_dict(state)
            epoch_before = saved['epoch']
            best = saved['best_prec1']
        else:
            print('>Warning: Could not read checkpoint! start fresh!!')

    loss = nn.MSELoss()
    opt = torch.optim.Adam(m.parameters(), 
                        world.lr, 
                        # momentum=world.momentum, 
                        weight_decay=world.weight_decay)
    if world.useCuda:
        m = m.cuda()
        loss = loss.cuda()
    # m.eval()
    m = m.float()
    if args.tensorboard:
        print(">ASK FOR tensorboard")
        print("\'tensorboard --logdir=runs\' to run it")
        with SummaryWriter(comment="-" + args.comment) as w:
            w.add_image("image", img, 0)
            for epoch in range(world.epochs):
                utils.adjust_learning_rate(opt, epoch)
                # print(epoch)
                # print(len(opt.state_dict()["param_groups"]))
                utils.train(data_train, m, loss, opt, epoch, w)
                pred = utils.validate(data_test, m ,loss, epoch)
                w.add_scalar("Loss/test", pred, epoch)
                is_best = pred < best
                best_pred = min(pred, best)
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': m.state_dict(),
                    "best_prec1": best_pred, 
                }, is_best)
    else:
        for epoch in range(world.epochs):
                utils.adjust_learning_rate(opt, epoch)
                # print(epoch)
                # print(len(opt.state_dict()["param_groups"]))
                utils.train(data_train, m, loss, opt, epoch)
                pred = utils.validate(data_test, m ,loss, epoch)
                is_best = pred < best
                best_pred = min(pred, best)
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': m.state_dict(),
                    "best_prec1": best_pred, 
                }, is_best)



