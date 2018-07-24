import tensorflow as tf
import numpy as np
import time
from dataset import Dataset
from model import ModelFactory
import math

class Solver:
    def __init__(self, args):
        self.dataset = Dataset(args)
        self.model = ModelFactory.newModel(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
        print 'model', self.model
        self.epoch = args.epoch
        self.verbose = args.verbose
        self.adv = args.adv
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=0)
        self.sess.run(self.model.assign_image, feed_dict={self.model.init_image: self.dataset.emb_image})

        self.weight_dir = args.weight_dir + '/'
        self.load()

    def one_iteration(self, feed_dict):
        st = time.time()
        self.sess.run([self.model.optimizer], feed_dict=feed_dict)

    def one_epoch(self):
        generator = self.dataset.batch_generator()
        api = [self.model.user_input, self.model.pos_input, self.model.neg_input]
        while True:
            try:
                feed_dict = dict(zip(api, generator.next()))
                self.one_iteration(feed_dict)
            except Exception as e:
                #print e.message
                break


    def train(self):
        for i in range(self.epoch):
            if i % self.verbose == 0:
                self.test('epoch %d' % i)
                self.save(i)
            self.one_epoch()

        self.save(i)

    def _score(self, para):
        r,K = para
        hr = r < K
        if hr:
            ndcg = math.log(2) / math.log(r + 2)
        else:
            ndcg = 0
        return (hr, ndcg)

    def test(self, message):
        st = time.time()
        generator = self.dataset.test_generator()
        api = [self.model.user_input, self.model.pos_input]
        d = []
        while True:
            try:
                feed_dict = dict(zip(api, generator.next()))
                preds = self.sess.run(self.model.pos_pred, feed_dict=feed_dict)

                rank = np.sum(preds[1:] >= preds[0])
                d.append(rank)
            except Exception as e:
                #print type(e), e.message
                break
        score5 = np.mean(map(self._score, zip(d,[5] * len(d))), 0)
        score10 = np.mean(map(self._score, zip(d,[10] * len(d))), 0)
        score20 = np.mean(map(self._score, zip(d, [20] * len(d))), 0)

        print message, score5, score10, score20
        print 'evaluation cost', time.time() - st

    def load(self):
        params = np.load(self.weight_dir + 'best-vbpr.npy')
        self.sess.run([self.model.assign_P, self.model.assign_Q, self.model.phi.assign(params[2])],
            {self.model.init_emb_P: params[0], self.model.init_emb_Q: params[1]})
        print 'load new parameters from best-vbpr.npy'


    def save(self, step):

        params = self.sess.run(tf.trainable_variables())
        path = '%s%s-%d.npy'%(self.weight_dir,self.model.get_saver_name(),step)
        np.save(path, params)
        print 'params are saved to ',path
        return

    def log(self):
        pass
