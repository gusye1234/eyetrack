from load_data import dataloader
from model import ITrackerModel, ItrackerImageModel
import utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torch import nn
import world
from parser import args
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

def syn_world(args):
    world.batch_size = args.batch_size
    world.doLoad = args.doload
    world.comment = args.comment    
    world.tensorboard = args.tensorboard
    world.epochs = args.epochs
    world.base_lr = args.lr


if __name__ == "__main__":
    # load config
    args = args().parse_args()
    syn_world(args)
    # load data
    tran = transforms.Compose([utils.Scale() ,utils.ToTensor()])
    data = dataloader(transform=tran)
    data_train = DataLoader(data, batch_size=world.batch_size, shuffle=True, num_workers=world.workers)
    world.n_batch = len(data_train)
    data_test = dataloader(mode="test", transform=tran)
    data_test = DataLoader(data_test, batch_size=world.batch_size, shuffle=True, num_workers=world.workers)

    # show config and data sample
    data_sample = data[0]
    img = utils.make_sample(data_sample)
    utils.show_config()

    # initialize model and restore training process
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
            print(">START EPOCH AT", epoch_before)
            best = saved['best_prec1']
        else:
            print('>Warning: Could not read checkpoint! start fresh!!')

    # define loss and optimizer
    # loss = nn.MSELoss(size_average=True)
    loss = nn.L1Loss(size_average=True)
    opt = torch.optim.Adam(m.parameters(), 
                        world.base_lr, 
                        # momentum=world.momentum, 
                        weight_decay=world.weight_decay)
    # switch to GPU
    if world.useCuda:
        m = m.cuda()
        loss = loss.cuda()
    
    # data format
    m = m.float()

    early_stop = utils.EarlyStopping(delta=0)
    early_stop.best_score = best

    # training process
    if args.tensorboard:
        print(">ASK FOR tensorboard")
        print("use command \'tensorboard --logdir=runs\' to run it")
        with SummaryWriter(comment="-" + args.comment) as w:
            w.add_image("image", img, 0)
            for epoch in range(world.epochs):
                utils.adjust_learning_rate(opt, epoch)
                # print(epoch)
                # print(len(opt.state_dict()["param_groups"]))
                utils.train(data_train, m, loss, opt, epoch, w)
                pred = utils.validate(data_test, m ,loss, epoch)
                w.add_scalar(f"Loss/test{world.base_lr}", pred, epoch)
                # is_best = pred < best
                # best = min(pred, best)
                # utils.save_checkpoint({
                #     'epoch': epoch + 1,
                #     'state_dict': m.state_dict(),
                #     "best_prec1": pred, 
                # }, is_best)
                early_stop(pred, m, epoch)
                if early_stop.early_stop:
                    print(">DONE")
                    break
    else:
        for epoch in range(epoch_before, world.epochs):
                utils.adjust_learning_rate(opt, epoch)
                utils.train(data_train, m, loss, opt, epoch)
                pred = utils.validate(data_test, m ,loss, epoch)
                pred = 10
                early_stop(pred, m, epoch)
                if early_stop.early_stop:
                    print(">DONE")
                    break



