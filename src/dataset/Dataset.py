import numpy as np
import random


class Dataset:
    def __init__(self, args):
        path = args.dataset
        self.f_feature = path + ('/f_%s.npy' % args.cnn)
        self.f_pos = path + '/pos.txt'
        self.f_neg = path + '/neg.txt'
        self.bsz = args.batch_size
        self.preload()

    def preload(self):
        self.emb_image = np.load(self.f_feature)
        self.fsz = self.emb_image.shape[1]
        self.pos = np.loadtxt(self.f_pos, dtype=np.int)
        self.usz, self.isz = np.max(self.pos, 0) + 1
        self.neg = np.loadtxt(self.f_neg, dtype=np.int)

        self.emb_image = self.emb_image / np.max(np.abs(self.emb_image))

        if self.usz < self.neg.shape[0]:
            self.usz = self.neg.shape[0]
        max_iid = self.neg.max()
        if self.isz <= max_iid:
            self.isz = max_iid + 1


        self.coldstart = set(self.neg[:,0].tolist()) - set(self.pos[:,1].tolist())
        self.pos = list(self.pos)

        self.inter = {}
        for u,i in self.pos:
            if u not in self.inter:
                self.inter[u] = set([])
            self.inter[u].add(i)

        print 'self.emb_image', self.emb_image.shape, \
              'self.pos.size', len(self.pos), \
              'self.neg.size', len(self.neg)
        print 'size', (self.usz, self.isz)

    def shuffle(self):
        random.shuffle(self.pos)

    def sample(self, p):
        u,i = self.pos[p]
        i_neg = i
        while i_neg in self.inter[u] or i_neg in self.coldstart:  # remove the cold start items from negative samples
            i_neg = random.randrange(self.isz)
        return (u,i,i_neg)

    def batch_generator(self):
        self.shuffle()
        sz = len(self.pos)

        for st in range(0, sz, self.bsz):
            samples = zip(*map(self.run, range(st, st + self.bsz)))
            yield map(np.array, samples)

    def test_generator(self):

        for u in range(0, self.usz):
            samples = zip(*[(u, i) for i in self.neg[u, :]])
            yield map(np.array,samples)

    def run(self, p):
        return self.sample(p)

