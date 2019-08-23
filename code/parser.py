import argparse
from torch import cuda



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args():
    parser = argparse.ArgumentParser(description="eye-tracker-model.")
    parser.add_argument("--tensorboard", help="ask if store the output to tensorboard",
                        type=str2bool, default=False)
    parser.add_argument("--comment", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default= cuda.device_count()*100 if cuda.device_count() != 0 else 4)
    parser.add_argument("--doload", type=str2bool, help="load previous weights or not", default=True)
    parser.add_argument("--epochs", type=int, default=100, help="traing total epochs")
    return parser