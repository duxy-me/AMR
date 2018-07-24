import argparse
from solver import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run RSGAN.")
    parser.add_argument('--dataset', nargs='?', default='/home/dxy/PycharmProjects/VCycle/venv/data/pin_tiny',
                        help='dataset path')
    parser.add_argument('--model', nargs='?', default='AMR',
                        help='model: AIR, BPR, VBPR, DUIF, Cycle, CycleOri, POP, AMR')
    parser.add_argument('--emb1_K', type=int, default=64, help='size of embeddings')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--lr', nargs='?', default='0.01', help='learning rate')
    parser.add_argument('--verbose', type=int, default=50, help='verbose')
    parser.add_argument('--epoch', type=int, default=500, help='epochs')
    parser.add_argument('--regs', nargs='?', default='0.000001', help='lambdas for regularization')
    parser.add_argument('--lmd', type=float, default=1, help='lambda for balance the common loss and adversarial loss')
    parser.add_argument('--keep_prob', type=float, default=0.6, help='keep probability of dropout layers')
    parser.add_argument('--adv', type=bool, default=False, help='adversarial training')
    parser.add_argument('--adv_type', nargs='?', default='grad', help='adversarial training type: grad, sign, rand')
    parser.add_argument('--cnn', nargs='?', default='resnet', help='cnn type: resnet, alexnet')
    parser.add_argument('--epsilon', type=float, default=2, help='epsilon for adversarial')
    parser.add_argument('--weight_dir', nargs='?', default='./weights', help='directory to store the weights')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print args

    print 'starting common Solver'
    s = Solver(args)

    s.train()
    s.test('final')
