from load_data import dataloader
from model import ITrackerModel, ItrackerImageModel
import utils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from torch import nn
import world
from parser import args
import time
from eval import eval, generate_heatmap
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


def syn_world(args):
    world.batch_size = args.batch_size
    world.doLoad = args.doload
    world.comment = args.tag
    world.tensorboard = args.tensorboard
    world.epochs = args.epochs
    world.base_lr = args.lr
    world.useSigmoid = args.sigmoid
    world.weight_file = args.weights
    world.resize = args.resize
    world.collect = args.collect
    world.activation = args.activation
    if world.activation not in ["tanh", "sigmoid", "none"]:
        raise TypeError("Please choose a activation function in [\"tanh\", \"sigmoid\", \"none\"]")
    if args.eval and args.weights == "":
        raise IOError("You must choose a pretrained-weight file to start eval mode")
    if args.eval == False and args.collect:
        raise TypeError("Can't collect intermediate data without eval mode opened")

if __name__ == "__main__":
    # load config
    args = args().parse_args()
    syn_world(args)
    if world.verbose:
        print("loaded configs")
    if world.useSigmoid:
        world.filename = "checkpoint_sigmoid.pth.tar"
    args.tag = time.strftime("%m-%d-%H:%M") + args.opt + str(args.lr) + "-" + world.comment + "_"
    world.filename = args.tag + world.filename

    # load data
    tran = transforms.Compose([utils.Scale(), utils.ToTensor()])
    data = dataloader(transform=tran)
    if len(data) == 0:
        print("Didn't find dataset.")
        raise ValueError("empty dataset")
    data_train = DataLoader(data, batch_size=world.batch_size, shuffle=True, num_workers=world.workers)
    world.n_batch = len(data_train)
    if args.eval == False:
        data_test = dataloader(mode="test", transform=tran)
        data_test = DataLoader(data_test, batch_size=world.batch_size, shuffle=True, num_workers=world.workers)
    else:
        data_test = dataloader(mode="test", transform=tran, folder=args.evalFolder)
        data_test = DataLoader(data_test, batch_size=world.batch_size, shuffle=False, num_workers=world.workers)
        print(">EVAL FOLDER:", args.evalFolder)
    if world.verbose:
        print("loaded data")

    # sample data and show configs
    data_sample = data[0]
    img = utils.make_sample(data_sample)
    utils.show_config()

    # initialize model and restore training process
    m = ITrackerModel()
    if world.verbose:
        print("initialized model")
    epoch_before = 0
    best = 1e10
    if world.doLoad:
        saved = utils.load_checkpoint(world.weight_file)
        if saved:
            print(
                '>Loading checkpoint for epoch %05d with loss %.5f ...' % (
                saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                m.module.load_state_dict(state)
            except:
                m.load_state_dict(state)
            epoch_before = saved['epoch']
            print(">START EPOCH AT", epoch_before)
            # best = saved['best_prec1']
        else:
            print('>Warning: Could not read checkpoint! start fresh!!')

    # define loss and optimizer
    # loss = nn.MSELoss(reduction='mean')
    # loss = nn.L1Loss(reduction='mean')
    loss = utils.L2loss()
    if args.opt == "adam":
        opt = torch.optim.Adam(m.parameters(),
                           world.base_lr,
                           # momentum=world.momentum,
                           weight_decay=world.weight_decay)
    elif args.opt == "SGD":
        opt = torch.optim.SGD(m.parameters(), 
                            world.base_lr,
                            momentum=world.momentum,
                            weight_decay = world.weight_decay)
    else:
        raise TypeError("Didn't support", args.opt, "optimizer! please choose one in [adam, SGD]")
    print(">USING", args.opt)
    # switch to GPU
    if world.useCuda:
        m = m.cuda()
        loss = loss.cuda()

    # data format
    m = m.float()
    if args.eval == False:
        early_stop = utils.EarlyStopping(delta=args.delta)
        early_stop.best_score = best

        # training process
        if args.tensorboard:
            print(">ASK FOR tensorboard")
            print("use command \'tensorboard --logdir=runs\' to run it")
            with SummaryWriter("/output/"+ "runs/"+time.strftime("%m-%d-%H:%M:%S-") + args.opt + str(args.lr) + "-" + world.comment) as w:
                w.add_image("image", img, 0)
                for epoch in range(world.epochs):
                    utils.adjust_learning_rate(opt, epoch)
                    # print(epoch)
                    # print(len(opt.state_dict()["param_groups"]))
                    utils.train(data_train, m, loss, opt, epoch, w)
                    pred = utils.validate(data_test, m, loss, epoch)
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
                pred = utils.validate(data_test, m, loss, epoch)
                print("predict is :", pred)
                early_stop(pred, m, epoch)
                if early_stop.early_stop:
                    print(">DONE")
                    break
    else:
        if args.generating:
            generate_heatmap(data_test, m, loss, args.evalFolder)
        else:
            eval(test_loader=data_test, model=m, criterion=loss)
        if world.collect:
            np.save("/output/middle.npy", np.concatenate(m.collect, axis=0))
        