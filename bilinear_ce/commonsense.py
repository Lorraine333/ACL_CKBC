import pdb
import sys
import time
import utils
import theano
import lasagne
import cPickle
import numpy as np
import random as random
from theano import tensor as T
from theano import config
from random import choice
from random import randint
from collections import OrderedDict
from utils import convertToIndex
from evaluate import evaluate_lstm
from lasagne.regularization import l2,regularize_network_params

class theano_word_model(object):

    #takes list of seqs, puts them in a matrix
    #returns matrix of seqs and mask    
    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen, 1)).astype(theano.config.floatX)
        x_len = np.zeros((n_samples,1)).astype('int32')
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x[idx, lengths[idx]:] = len(self.words)-1
            x_len[idx,:] = lengths[idx]
            x_mask[idx, :lengths[idx]] = 1.
        return x, x_mask, x_len

    def saveParams(self, fname):
        print 'save model to',fname
        f = file(fname, 'wb')
        save_model = {}
        save_model['words_name'] = self.words
        save_model['rel_name'] = self.rel
        save_model['embeddings'] = self.getWe()
        save_model['rel'] = self.getRel()
        save_model['weight'] = self.getWeight()
        save_model['bias'] = self.getBias()
        cPickle.dump(save_model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)


    def getpairs(self, batch, params):
        relsize = self.relsize

        Rel = self.getRel()
        we = self.getWe()
        Rel0 = np.reshape(Rel,(-1,relsize))
        newd = [convertToIndex(i, self.words, we, self.rel, Rel0) for i in batch]
        g1=[];g2=[];R=[]
        length = len(batch)
        for idx, e in enumerate(newd):
            (r, t1, t2, s) =e
            g1.append(t1)
            g2.append(t2)
            R.append(r)
        
        #batch is list of tuples
        g1x, g1mask, g1length= self.prepare_data(g1)
        g2x, g2mask, g2length = self.prepare_data(g2)
        embg1 = self.feedforward_function(g1x,g1mask,g1length)
        embg2 = self.feedforward_function(g2x,g2mask,g2length)

        p1 = []
        p2 = []
        neg_r = []
        best = best1 = best2 = 1
        if(params.type == 'MAX'):
            for i in range(length):
                id0 = R[i]
                min0 = -5000
                min1 = -5000
                min2 = -5000
                vec1 = embg1[i,:]
                vec2 = embg2[i,:]
                vec_r = Rel[id0,:,:]
                for j in range(length):
                    if j != i:
                        gv1=embg1[j,:]
                        temp1 = np.dot(gv1, vec_r)
                        np1 = np.inner(temp1,vec2)
                        if np1 > min0:
                            min0=np1
                            best=j
                for j1 in range(length):
                    if j1 != i:
                        gv2=embg2[j1,:]
                        temp11 = np.dot(vec1, vec_r)
                        np11 = np.inner(temp1,gv2)
                        if np11 > min1:
                            min1=np11
                            best1=j1
                for j2 in range(length):
                    if j2 != i:
                        id1 = R[j2]
                        matrix_r = Rel[id1,:,:]
                        temp111 = np.dot(vec1, matrix_r)
                        np111 = np.inner(temp111, vec2)
                        if np111 > min2:
                            min2=np111
                            best2 = j2
                p1.append(g1[best])
                p2.append(g2[best1])
                neg_r.append(R[best2])

        if(params.type == 'MIX'):
            for i in range(length):
                r1 = randint(0,1)
                if r1 == 1:
                    id0 = R[i]
                    min0 = -5000
                    min1 = -5000
                    min2 = -5000
                    vec1 = embg1[i,:]
                    vec2 = embg2[i,:]
                    vec_r = Rel[id0,:,:]
                    for j in range(length):
                        if j != i:
                            gv1=embg1[j,:]
                            temp1 = np.dot(gv1, vec_r)
                            np1 = np.inner(temp1,vec2)
                            if np1 > min0:
                                min0=np1
                                best=j
                    for j1 in range(length):
                        if j1 != i:
                            gv2=embg2[j1,:]
                            temp11 = np.dot(vec1, vec_r)
                            np11 = np.inner(temp1,gv2)
                            if np11 > min1:
                                min1=np11
                                best1=j1
                    for j2 in range(length):
                        if j2 != i:
                            id1 = R[j2]
                            matrix_r = Rel[id1,:,:]
                            temp111 = np.dot(vec1, matrix_r)
                            np111 = np.inner(temp111, vec2)
                            if np111 > min2:
                                min2=np111
                                best2 = j2
                    p1.append(g1[best])
                    p2.append(g2[best1])
                    neg_r.append(R[best2])
                else:
                    id0 = R[i]
                    wpick=['','','']
                    while(wpick[0]==''):
                        index=random.randint(0,len(g1)-1)
                        if(index!=i):
                            wpick[0]=g1[index]
                            p1.append(wpick[0])

                    while(wpick[1]==''):
                        index=random.randint(0,len(g2)-1)
                        if(index!=i):
                            wpick[1]=g2[index]
                            p2.append(wpick[1])

                    while(wpick[2]==''):
                        index=random.randint(0,len(R)-1)
                        if(index!=i):
                            wpick[2]=R[index]
                            neg_r.append(wpick[2])

        if(params.type == 'RAND'):
            for i in range(length):
                id0 = R[i]
                wpick=['','','']
                while(wpick[0]==''):
                    index=random.randint(0,len(g1)-1)
                    if(index!=i):
                        wpick[0]=g1[index]
                        p1.append(wpick[0])

                while(wpick[1]==''):
                    index=random.randint(0,len(g2)-1)
                    if(index!=i):
                        wpick[1]=g2[index]
                        p2.append(wpick[1])

                while(wpick[2]==''):
                    index=random.randint(0,len(R)-1)
                    if(index!=i):
                        wpick[2]=R[index]
                        neg_r.append(wpick[2])
        
        p1x, p1mask, p1length= self.prepare_data(p1)
        p2x, p2mask, p2length = self.prepare_data(p2)

        return (g1x,g1mask,g1length,g2x,g2mask,g2length,p1x,p1mask,p1length,p2x,p2mask,p2length,R,neg_r)

    def __init__(self, We_initial, words, memsize, rel, relsize, Rel_init,  LC, LW, eta, margin, usepeep, acti):

        self.LC = LC
        self.LW = LW
        self.margin = margin
        self.memsize = memsize
        self.usepeep = usepeep
        self.relsize = relsize
        self.words = words
        self.rel = rel  

        self.a1 = np.zeros((35,relsize, relsize))
        for k in range(35):
            for i in range(self.a1.shape[1]):
                for j in range(self.a1.shape[2]):
                    if(i==j):
                            self.a1[k][i][j] = 1
                    else:
                            self.a1[k][i][j] = 0

        self.Rel = theano.shared(Rel_init).astype(theano.config.floatX)
        self.iden = theano.shared(self.a1)
        self.we = theano.shared(We_initial).astype(theano.config.floatX)
        
        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        p1batchindices = T.imatrix(); p2batchindices = T.imatrix()
        g1mask = T.tensor3(); g2mask = T.tensor3()
        p1mask = T.tensor3(); p2mask = T.tensor3()
        g1length = T.imatrix(); g2length = T.imatrix()
        p1length = T.imatrix(); p2length = T.imatrix()
        target=T.dmatrix()

        g1mask = T.patternbroadcast(g1mask,broadcastable = [False, False, True])
        g2mask = T.patternbroadcast(g2mask,broadcastable = [False, False, True])
        p1mask = T.patternbroadcast(p1mask,broadcastable = [False, False, True])
        p2mask = T.patternbroadcast(p2mask,broadcastable = [False, False, True])

        We0 = T.dmatrix()
        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))

        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.we.get_value().shape[0], output_size=self.we.get_value().shape[1], W=self.we)
        l_out = l_emb
        embg1 = lasagne.layers.get_output(l_emb, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_emb, {l_in:g2batchindices, l_mask:g2mask})
        embp1 = lasagne.layers.get_output(l_emb, {l_in:p1batchindices, l_mask:p1mask})
        embp2 = lasagne.layers.get_output(l_emb, {l_in:p2batchindices, l_mask:p2mask})

        embg1 = embg1 * g1mask
        embg1_sum = T.sum(embg1, axis = 1)
        embg1_len = T.patternbroadcast(g1length,broadcastable = [False, True])
        embg1_mean = embg1_sum/embg1_len
        embg1_mean = embg1_mean.reshape([-1,self.memsize])


        embg2 = embg2 * g2mask
        embg2_sum = T.sum(embg2, axis = 1)
        embg2_len = T.patternbroadcast(g2length,broadcastable = [False, True])
        embg2_mean = embg2_sum/embg2_len
        embg2_mean = embg2_mean.reshape([-1,self.memsize])

        embp1 = embp1 * p1mask
        embp1_sum = T.sum(embp1, axis = 1)
        embp1_len = T.patternbroadcast(p1length,broadcastable = [False, True])
        embp1_mean = embp1_sum/embp1_len
        embp1_mean = embp1_mean.reshape([-1,self.memsize])

        embp2 = embp2 * p2mask
        embp2_sum = T.sum(embp2, axis = 1)
        embp2_len = T.patternbroadcast(p2length,broadcastable = [False, True])
        embp2_mean = embp2_sum/embp2_len
        embp2_mean = embp2_mean.reshape([-1,self.memsize])


        #############################################################
        r=T.ivector()
        p3=T.ivector()
        r0=self.Rel[r]
        r1=self.Rel[p3]

        self.a2 = np.random.uniform(low = -0.2, high = 0.2, size = [memsize,relsize])
        self.a3 = np.random.uniform(low = -0.2, high = 0.2, size = [relsize,])
        self.w = theano.shared(self.a2)
        self.b = theano.shared(self.a3)

        embg1_rel = T.tanh(T.dot(embg1_mean,self.w)+self.b.dimshuffle('x', 0))
        embg2_rel = T.tanh(T.dot(embg2_mean,self.w)+self.b.dimshuffle('x', 0))
        embp1_rel = T.tanh(T.dot(embp1_mean,self.w)+self.b.dimshuffle('x', 0))
        embp2_rel = T.tanh(T.dot(embp2_mean,self.w)+self.b.dimshuffle('x', 0))

        g1g2 = T.batched_dot(embg1_rel,r0)
        g1g2 = T.batched_dot(g1g2,embg2_rel)
        p1g1_neg = T.batched_dot(embp1_rel,r0)
        p1g1_neg = T.batched_dot(p1g1_neg,embg2_rel)
        p2g2_neg = T.batched_dot(embg1_rel,r0)
        p2g2_neg = T.batched_dot(p2g2_neg,embp2_rel)
        g1g2_neg = T.batched_dot(embg1_rel,r1)
        g1g2_neg = T.batched_dot(g1g2_neg,embg2_rel)

        g1g2 = T.nnet.sigmoid(g1g2).reshape([-1,1])
        p1g1_neg = T.nnet.sigmoid(p1g1_neg).reshape([-1,1])
        p2g2_neg = T.nnet.sigmoid(p2g2_neg).reshape([-1,1])
        g1g2_neg = T.nnet.sigmoid(g1g2_neg).reshape([-1,1])

        lsm=T.concatenate([g1g2,p1g1_neg,p2g2_neg,g1g2_neg],axis=0)

        #updates
        network_params = lasagne.layers.get_all_params(l_out, trainable=True)
        network_params.append(self.w)
        network_params.append(self.b)
        self.all_params = network_params
        self.all_params.append(self.Rel)
        self.all_params.append(self.we)

        #feedforward
        self.feedforward_function = theano.function([g1batchindices,g1mask,g1length], embg1_rel, on_unused_input='warn')
        # self.softMax=theano.function([ar],outputs=softmaxOutput2)
        #cost_function
        softmaxOutput=lsm.clip(1e-7,1.0 - 1e-7)
        # softmaxOutput2=lsm2.clip(1e-7,1.0 - 1e-7)

        loss=lasagne.objectives.binary_crossentropy(softmaxOutput,target)
        loss=lasagne.objectives.aggregate(loss,mode='mean')

        l2_penalty1=lasagne.regularization.apply_penalty(network_params,l2)

        cost_new = (1000*loss) +(self.LC * l2_penalty1) + (self.LW * lasagne.regularization.l2(r0 - self.iden[r])) 

        #train_function
        updates = lasagne.updates.adagrad(cost_new, self.all_params, learning_rate = eta)
        self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices,p2batchindices,
                             g1mask, g2mask, p1mask,p2mask,g1length,g2length,p1length,p2length,r, We0, p3, target], [cost_new, loss], updates=updates, on_unused_input='warn')

        #############################################################
    def GetVector(self, g1batchindices, g1mask,g1length):
        return self.feedforward_function(g1batchindices, g1mask,g1length)

    def getWe(self):
        return self.we.get_value()

    def getRel(self):
        return self.Rel.get_value()

    def getWeight(self):
        return self.w.get_value()
    
    def getBias(self):
        return self.b.get_value()

    def evaluate(self):
        Rel = self.getRel()
        Rel0 = np.reshape(Rel,(-1,self.relsize))
        We =  self.getWe()
        return evaluate_lstm(self, We, self.words, Rel0, self.rel, self.relsize)

    #trains parameters
    def train(self, data, params, We0):
        start_time = time.time()

        counter = 0
        try:
            COPA = self.evaluate()
            # print 'COPA Result: ', COPA
            for eidx in xrange(params.epochs):
                t1 = time.time()
                n_samples = 0
                # Get new shuffled index for the training set.
                kf = self.get_minibatches_idx(len(data), params.batchsize, shuffle=True)
                uidx = 0
                for _, train_index in kf:
                    uidx += 1
                    batch = [data[t] for t in train_index]
                    n_samples += len(batch)
                    (g1x,g1mask,g1length,g2x,g2mask,g2length,p1x,p1mask,p1length,p2x,p2mask,p2length,r,neg_r) = self.getpairs(batch, params)
                    #get targets
                    targetg1g2=(np.ones((len(batch),1),dtype=np.float))
                    targetp1g2=(np.zeros((len(batch),1),dtype=np.float))
                    targetg1p2=(np.zeros((len(batch),1),dtype=np.float))
                    targetg1r1g2=(np.zeros((len(batch),1),dtype=np.float))

                    target=np.concatenate([targetg1g2,targetp1g2,targetg1p2,targetg1r1g2],axis=0)

                    cost, loss = self.train_function(g1x, g2x, p1x, p2x, g1mask, g2mask, p1mask,p2mask,g1length,g2length,p1length,p2length,r,We0,neg_r,target)

                    

                print 'Epoch ', (eidx+1), 'Cost ', cost



                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'

                COPA = self.evaluate()
                # print 'COPA Result: ', COPA
                t2 = time.time()
                print "Cost time: "+str(t2-t1)

                if(params.save):
                    counter += 1
                    self.saveParams(params.outfile+str(counter)+'.pickle')
                
                print 'Seen %d samples' % n_samples
                print ' '
        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.time()
        print "total time:", (end_time - start_time)

