import theano
import numpy as np
from theano import tensor as T
from theano import config
import pdb
import random as random
from utils import Relu
from utils import Sigmoid
import time
import utils
from collections import OrderedDict
import lasagne
import sys
import cPickle
from utils import convertToIndex
from utils import lookupwordID
from evaluate import evaluate_lstm
from utils import ReluT
from lasagne.layers import get_output
from lasagne.regularization import l2,regularize_network_params
from random import choice

class theano_word_model(object):

    #takes list of seqs, puts them in a matrix
    #returns matrix of seqs and mask    
    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
        for idx, s in enumerate(list_of_seqs):
        #print maxlen
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        return x, x_mask

    def saveParams(self, fname):
        f = file(fname, 'wb')
        cPickle.dump(self.all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
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

    def getpairs2(self, batch, params):
        embed_size = self.memsize
        Rel = self.getRel()
        we = self.getWe()
        # Rel0 = np.reshape(Rel,(-1,relsize))
        newd = [convertToIndex(i, self.words, we, self.rel, Rel) for i in batch]
        g1=[];g2=[];R=[]
        #print newd
        length = len(batch)

        for idx, e in enumerate(newd):
            (r, t1, t2, s) =e
            g1.append(t1)
            g2.append(t2)
            R.append(r)
        #batch is list of tuples

        p11=[]
        p22=[]
        p3=[]
        if(params.type == 'MAX'):
            for i in range(length):
                #print 'i: ',i
                id0 = R[i]
                wpick=['','','']
                while(wpick[0]==''):
                    index=random.randint(0,len(g1)-1)
                    if(index!=i):
                        wpick[0]=g1[index]
                        p11.append(wpick[0])

                while(wpick[1]==''):
                    index=random.randint(0,len(g2)-1)
                    if(index!=i):
                        wpick[1]=g2[index]
                        p22.append(wpick[1])

                while(wpick[2]==''):
                    index=random.randint(0,len(R)-1)
                    if(index!=i):
                        wpick[2]=R[index]
                        p3.append(wpick[2])

        delim= (lookupwordID(we, self.words, "#"))

        pT=[a+delim+b for a,b in zip(g1,g2)]
        pTuple, pTupleMask=self.prepare_data(pT)
        neT1=[a+delim+b for a,b in zip(p11,g2)]
        neTuple1, neTuple1Mask=self.prepare_data(neT1)
        neT2=[a+delim+b for a,b in zip(g1,p22)]
        neTuple2, neTuple2Mask=self.prepare_data(neT2)


        return (R,p3,pTuple,pTupleMask,neTuple1,neTuple1Mask,neTuple2,neTuple2Mask)


    def __init__(self, We_initial, words, layersize, embed_size, rel, batchsize, Rel_init, LC, eta, margin, usepeep, fin,initiallization,relsize,activation):

        self.LC = LC
        self.margin = margin
        self.memsize = embed_size
        self.usepeep = usepeep
        self.batchsize = batchsize
        self.words = words
        self.rel = rel  
        self.Rel = theano.shared(Rel_init).astype(theano.config.floatX)
        self.we = theano.shared(We_initial).astype(theano.config.floatX)
        self.fin=fin
        self.initiallization=initiallization
        self.relsize=relsize
        self.activation=activation

        #symbolic params

        target=T.dmatrix()
        pTuple=T.imatrix();neTuple1=T.imatrix();neTuple2=T.imatrix()
        pTupleMask=T.matrix();neTuple1Mask=T.matrix(); neTuple2Mask=T.matrix()

        self.eta = T.dscalar()
        self.lam = T.dscalar()
        g1len=T.iscalar();g2len=T.iscalar(); p1len=T.iscalar(); p2len=T.iscalar(); poolSize=T.iscalar()
        poolSize=1
        
        We0 = T.dmatrix()
        #get embeddings
        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        # l_rel = lasagne.layers.InputLayer(shape = (None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.we.get_value().shape[0], output_size=self.we.get_value().shape[1], W=self.we)
        l_lstm1 = lasagne.layers.LSTMLayer(l_emb, layersize, peepholes=True, grad_clipping=100, mask_input = l_mask)
        l_lstm2 = lasagne.layers.LSTMLayer(l_emb, layersize, peepholes=True, grad_clipping=100, mask_input = l_mask, backwards=True)
        l_lstm3 =  lasagne.layers.ConcatLayer([l_lstm1, l_lstm2], axis=2)


        layer_poolingPo=lasagne.layers.get_output(l_lstm3,{l_in:pTuple, l_mask:pTupleMask})
        layer_poolingNe1=lasagne.layers.get_output(l_lstm3,{l_in:neTuple1, l_mask:neTuple1Mask})
        layer_poolingNe2=lasagne.layers.get_output(l_lstm3,{l_in:neTuple2, l_mask:neTuple2Mask})


        l_poolingPo=layer_poolingPo.max(axis=1)
        l_poolingNe1=layer_poolingNe1.max(axis=1)
        l_poolingNe2=layer_poolingNe2.max(axis=1)

        embPo = l_poolingPo.reshape([-1,self.memsize])
        embNe1 = l_poolingNe1.reshape([-1,self.memsize])
        embNe2 = l_poolingNe2.reshape([-1,self.memsize])






#############################################################
        r=T.ivector()
        p3=T.ivector()
        r0=self.Rel[r]
        r1=self.Rel[p3]

        input_vec = T.concatenate([embPo, r0], axis = 1)
        input_vec_neg = T.concatenate([embNe1,r0],axis = 1)
        input_vec_neg1 = T.concatenate([embNe2,r0], axis = 1)
        input_vec_neg2 = T.concatenate([embPo,r1], axis = 1)


        ar=T.dmatrix()

        vecAynaz=T.concatenate([input_vec,input_vec_neg,input_vec_neg1,input_vec_neg2],axis=0)



        l_in1 = lasagne.layers.InputLayer(shape=(None,(self.memsize)+self.relsize))

        if(self.activation=='none'):
            denseLayer1=lasagne.layers.DenseLayer(l_in1,num_units=600,W=lasagne.init.Normal(),nonlinearity=None)
        if(self.activation=='sigmoid'):
            denseLayer1=lasagne.layers.DenseLayer(l_in1,num_units=600,W=lasagne.init.Normal(),nonlinearity=lasagne.nonlinearities.sigmoid)
        if(self.activation=='rectify'):
            denseLayer1=lasagne.layers.DenseLayer(l_in1,num_units=800,W=lasagne.init.Normal(),nonlinearity=lasagne.nonlinearities.rectify)

        denseLayer2=lasagne.layers.DenseLayer(denseLayer1,num_units=200,W=lasagne.init.Normal(),nonlinearity=lasagne.nonlinearities.rectify)

        denseLayer=lasagne.layers.DenseLayer(denseLayer2,num_units=1,W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.sigmoid)


        lsm=get_output(denseLayer,{l_in1:vecAynaz})
        lsm2=get_output(denseLayer,{l_in1:ar})

        softmaxOutput=lsm
        softmaxOutput2=lsm2


        loss=lasagne.objectives.binary_crossentropy(softmaxOutput,target)
        loss=lasagne.objectives.aggregate(loss,mode='mean')
        network_params1 = lasagne.layers.get_all_params([denseLayer,l_lstm3,l_lstm2,l_lstm1],trainable=True)

        print network_params1
        self.all_params = network_params1
        self.all_params.append(self.Rel)
        self.all_params.append(self.we)

        self.feedforward_function = theano.function([pTuple,pTupleMask], embPo)

        l2_penalty1=lasagne.regularization.apply_penalty(network_params1,l2)

        cost_new = (1000*loss) +(self.LC * l2_penalty1)


        updates = lasagne.updates.adagrad(cost_new, self.all_params, eta)


        self.softMax=theano.function([ar],outputs=softmaxOutput2)

        self.train_function1 = theano.function(inputs = [r,p3,target,pTuple,pTupleMask,neTuple1,neTuple1Mask,neTuple2,neTuple2Mask], outputs = [cost_new,loss], updates=updates, on_unused_input='warn')


###########################################################
        
    def GetVector(self, g1batchindices, g1mask):
        return self.feedforward_function(g1batchindices, g1mask)

    def GetVectorNew(self,in_vector):
        return self.softMax(in_vector)

    def log_softmax(self,x):
        xdev = x - x.max(1, keepdims=True)
        return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

    def crossentropy_logdomain(log_predictions, targets):
        return -T.sum(targets * log_predictions, axis=1)

    def getWe(self):
        return self.we.get_value()

    def getRel(self):
        return self.Rel.get_value()

    def getW1(self):
        return self.w1.get_value()
    
    def getB1(self):
        return self.b1.get_value()
    
    def getW2(self):
        return self.w2.get_value()
    
    def getB2(self):
        return self.b2.get_value()
    
    def evaluate(self):
        Rel = self.getRel()
        # Rel0 = np.reshape(Rel,(-1,self.memsize))
        We =  self.getWe()
        return evaluate_lstm(self, We, self.words, Rel, self.rel,self.memsize, self.relsize,self.fin)

    #trains parameters
    def train(self, data, params, We0,fin):
        start_time = time.time()
        #evaluate_all(self,words)
        COPA = self.evaluate()
            #print COPA
        print 'COPA Result: ', COPA
        fin.write("COPA Result: "+ str(COPA)+"\n")
        # print 'WSC Result', wsc
        print ' '
        i=1
        counter = 0
        try:
            for eidx in xrange(params.epochs):
                epoch_start=time.time()
                n_samples = 0
                i=1
                # Get new shuffled index for the training set.
                kf = self.get_minibatches_idx(len(data), params.batchsize, shuffle=True)
                uidx = 0
                for _, train_index in kf:

                    uidx += 1

                    batch = [data[t] for t in train_index]
                    n_samples += len(batch)
                    (r,p3,pTuple,pTupleMask,neTuple1,neTuple1Mask,neTuple2,neTuple2Mask) = self.getpairs2(batch, params)
                    print "batch number: ", i
    

                    t1 = time.time()

                    targetg1g2=(np.ones((len(batch),1),dtype=np.float))
                    targetp1g2=(np.zeros((len(batch),1),dtype=np.float))
                    targetg1p2=(np.zeros((len(batch),1),dtype=np.float))
                    targetg1r1g2=(np.zeros((len(batch),1),dtype=np.float))

                    target=np.concatenate([targetg1g2,targetp1g2,targetg1p2,targetg1r1g2],axis=0)
                    cost,loss = self.train_function1(r,p3,target,pTuple,pTupleMask,neTuple1,neTuple1Mask,neTuple2,neTuple2Mask)
                    #print 'batch finished:', i
                    i=i+1

            #print costpg
            #cost1 = self.train_function1(g1x, g2x, p1x, g1mask, g2mask, p1mask,r)
                print 'Epoch ', (eidx+1), 'Cost ', cost, 'Cost no regularization ',loss
                fin.write('Epoch '+ str(eidx+1)+ 'Cost '+ str(cost) +"\n")
                t2 = time.time()
                #print "cost time: "+str(t2-t1)
        
                            
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    fin.write('NaN detected'+"\n")
           
                COPA = self.evaluate()
            #print COPA
                print 'COPA Result: ', COPA
                fin.write('COPA Result: '+ str(COPA)+"\n")
                # print 'WSC Result', wsc
                print ' '
             

                if(params.save):
                    counter += 1
                    self.saveParams(params.outfile+str(counter)+'.pickle')

                #evaluate_all(self,self.words)

                print 'Seen %d samples' % n_samples
                fin.write('Seen %d samples' % n_samples+"\n")

                epoch_end=time.time()
                print "time for epoch ", (eidx+1),": ", (epoch_end - epoch_start)
                fin.write("time for epoch "+ str(eidx+1)+": "+ str(epoch_end - epoch_start)+"\n")

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.time()
        print "total time:", (end_time - start_time)
        fin.write("total time:"+ str(end_time - start_time)+"\n")
