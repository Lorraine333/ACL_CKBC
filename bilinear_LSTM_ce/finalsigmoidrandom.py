
import theano
import numpy as np
from theano import tensor as T
from theano import config
import pdb
import random as random
#from evaluate import evaluate_all
import time
import utils
#from LSTMLayerNoOutput import LSTMLayerNoOutput
from collections import OrderedDict
import lasagne
import sys
import cPickle
from utils import convertToIndex
from evaluateall import evaluate_lstm

def checkIfQuarter(idx,n):
    #print idx, n
    if idx==round(n/4.) or idx==round(n/2.) or idx==round(3*n/4.):
        return True
    return False

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
	    #print s
	    #print maxlen
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        return x, x_mask

    def saveParams(self, fname):
        f = open(fname, 'w')
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

    def getpairs(self, batch, params):
	relsize = self.relsize

	Rel = self.getRel()
	we = self.getWe()
	Rel0 = np.reshape(Rel,(-1,relsize))
	newd = [convertToIndex(i, self.words, we, self.rel, Rel0) for i in batch]
	g1=[];g2=[];R=[]
	#print newd
	length = len(batch)
	#r0=np.zeros((length,relsize, relsize)).astype(theano.config.floatX)
	#print relsize
	for idx, e in enumerate(newd):
		(r, t1, t2, s) =e
		#print relsize
		#print length
		#print r
		g1.append(t1)
		g2.append(t2)
		R.append(r)
        #batch is list of tuples
        
     	g1x, g1mask= self.prepare_data(g1)
	#maxlen = g1x.shape[1]
	#print maxlen
     	g2x, g2mask = self.prepare_data(g2)


	p1=[]
     	for i in range(length):
		
		p1.append(g1[random.randint(0, length -1)])
		

        p1x, p1mask = self.prepare_data(p1)
	
	p2=[]
        for i in range(length):
                
                p2.append(g2[random.randint(0, length -1)])
                
	p2x, p2mask = self.prepare_data(p2)

	PR=[]
        for i in range(length):
                PR.append(R[random.randint(0, length -1)])
            
        return (g1x,g1mask,g2x,g2mask,p1x,p1mask, p2x, p2mask, R, PR)

#We, params.layersize, params.LC, params.LW,
#                                  params.updateword, params.eta, params.peephole, params.outgate)

    def __init__(self, We_initial, words, layersize, memsize, rel, relsize, Rel_init,  LC, LW, eta, margin, usepeep, updatewords):

        self.LC = LC
        self.LW = LW
        self.margin = margin
        self.layersize = layersize
        self.memsize = memsize
        self.usepeep = usepeep
	self.relsize = relsize
	self.words = words
	self.rel = rel	

        #sielf.a = np.zeros((35,relsize, relsize))
        #for k in range(35):
         #   for i in range(relsize):
          #  	for j in range(relsize):
           #     	if(i==j):
            #        		self.a[k][i][j] = 1+random.uniform(-0.2,0.2)
             #   	else:
              #      		self.a[k][i][j] = random.uniform(-0.2,0.2)

        #params
	#print self.a
	self.a1 = np.zeros((35,relsize, relsize))
	for k in range(35):
        	for i in range(relsize):
            		for j in range(relsize):
                		if(i==j):
                    			self.a1[k][i][j] = 1
                		else:
                    			self.a1[k][i][j] = 0

	self.Rel = theano.shared(Rel_init).astype(theano.config.floatX)
	#print self.Rel
        self.iden = theano.shared(self.a1)
        #initial_We = theano.shared(We_initial).astype(config.floatX)
	initial_We = theano.shared(We_initial).astype(config.floatX)
        self.we = theano.shared(We_initial).astype(config.floatX)
	#print self.we
        #symbolic params
        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        p1batchindices = T.imatrix(); p2batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix()
        p1mask = T.matrix(); p2mask = T.matrix()
	
	We0 = T.dmatrix()
        #get embeddings
        l_in = lasagne.layers.InputLayer((None, None, 1))
	
	batchsize, seqlen, _ = l_in.input_var.shape
	#print seqlen
	
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.we.get_value().shape[0], output_size=self.we.get_value().shape[1], W=self.we)
        l_lstm1 = lasagne.layers.LSTMLayer(l_emb, layersize, peepholes=usepeep, learn_init=False, mask_input = l_mask)
	l_lstm2 = lasagne.layers.LSTMLayer(l_emb, layersize, peepholes=usepeep, learn_init=False, mask_input = l_mask, backwards=True)
	l_out =  lasagne.layers.ConcatLayer([l_lstm1, l_lstm2], axis=2)

	l_out0 = lasagne.layers.SliceLayer(l_out, -1, 1)        

	

	embg1 = lasagne.layers.get_output(l_out0, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_out0, {l_in:g2batchindices, l_mask:g2mask})
        embp1 = lasagne.layers.get_output(l_out0, {l_in:p1batchindices, l_mask:p1mask})
        embp2 = lasagne.layers.get_output(l_out0, {l_in:p2batchindices, l_mask:p2mask})
	


	self.a2 = np.random.uniform(low = -0.01, high = 0.01, size = [2*memsize,relsize])
        self.a3 = np.random.uniform(low = -0.0, high = 0.0, size = [relsize,])
	self.w = theano.shared(self.a2)
        self.b = theano.shared(self.a3)
	

	embg1 = T.nnet.relu(T.dot(embg1,self.w)+self.b.dimshuffle('x', 0))
	embg2 = T.nnet.relu(T.dot(embg2,self.w)+self.b.dimshuffle('x', 0))
	embp1 = T.nnet.relu(T.dot(embp1,self.w)+self.b.dimshuffle('x', 0))
	embp2 = T.nnet.relu(T.dot(embp2,self.w)+self.b.dimshuffle('x', 0))

	r=T.ivector()
	pr = T.ivector()
	r0=self.Rel[r]
	pr0 =  self.Rel[pr]
        #objective function
        g1g2 = T.batched_dot(embg1,r0)
	g1g2 = T.batched_dot(g1g2,embg2)
        #g1g2 = 1- g1g2
	g1g2 = T.nnet.sigmoid(g1g2)
   
        p1g1 = T.batched_dot(embp1,r0)
	p1g1 = T.batched_dot(p1g1,embg2)
	p1g1 = T.nnet.sigmoid(p1g1)

	p2g2 = T.batched_dot(embg1,r0)
        p2g2 = T.batched_dot(p2g2,embp2)
	p2g2 = T.nnet.sigmoid(p2g2)

        
	cost1 =  T.log(g1g2)
	
	cost2 = T.log(1-p1g1) + T.log(1-p2g2)
	
	nr = T.batched_dot(embg1,pr0)
        nr = T.batched_dot(nr,embg2)
	nr = T.nnet.sigmoid(nr)
        #nr = g1g2 + nr
        #cost_r =  nr *(T.gt( nr , 0))
        #cost3 = T.mean(cost_r)
        cost3 = T.log(1-nr)
	cost_r = - cost1 - cost2 - cost3
	cost = T.mean(cost_r)
       


        network_params = lasagne.layers.get_all_params(l_out, trainable=True)
	network_params.append(self.w)
	network_params.append(self.b)
        
      

	
        if updatewords:
            word_reg = 100*self.LW*lasagne.regularization.l2(network_params[0] - initial_We)
            cost = cost + word_reg
	else :
	    word_reg =theano.shared(0)

	
	network_params.pop(0)
        l2 = self.LC*sum(lasagne.regularization.l2(x) for x in network_params)
   
	cost = cost + l2
        #feedforward
	
        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, p1batchindices,p2batchindices,
                             g1mask, g2mask, p1mask, p2mask, r, pr], cost, on_unused_input='warn')



        #updates
	self.all_params = lasagne.layers.get_all_params(l_out, trainable=True)
	self.all_params.append(self.w)
	self.all_params.append(self.b)
	self.all_params.append(self.Rel)

        self.train_function = None
        if updatewords:
            updates = lasagne.updates.adagrad(cost, self.all_params, eta)
            self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                             g1mask, g2mask, p1mask, p2mask, r, pr], [cost, word_reg], updates=updates, on_unused_input='warn')
        else:
            self.all_params = network_params
            self.all_params.append(self.Rel)
            updates = lasagne.updates.adagrad(cost, self.all_params, eta)
            self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                             g1mask, g2mask, p1mask, p2mask, r, pr], [cost, word_reg], updates=updates, on_unused_input='warn')


    def GetVector(self, g1batchindices, g1mask):
	return self.feedforward_function(g1batchindices, g1mask)

    def getWe(self):
        return self.we.get_value()



    def getRel(self):
        return self.Rel.get_value()
    
    def evaluate(self):
	Rel = self.getRel()
	Rel0 = np.reshape(Rel,(-1,self.relsize))
	We =  self.getWe()
	return evaluate_lstm(self, We, self.words, Rel0, self.rel, self.relsize)

    #trains parameters
    def train(self, data, params):
        start_time = time.time()
        #evaluate_all(self,words)
	text_file = open(params.outfile, "w")
	result0 =0
	result1 =0
        counter = 0
        try:
            for eidx in xrange(params.epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = self.get_minibatches_idx(len(data), params.batchsize, shuffle=True)
                uidx = 0
                for _, train_index in kf:

                    uidx += 1

                    batch = [data[t] for t in train_index]
		    n_samples += len(batch)

                    #(g1x,g1mask,g2x,g2mask,p1x,p1mask,p2x,p2mask) = self.getpairs(batch, params)
		    (g1x,g1mask,g2x,g2mask,p1x,p1mask,p2x,p2mask, r, pr) = self.getpairs(batch, params)
	             
 	

                    t1 = time.time()
                    cost, cost1 = self.train_function(g1x, g2x, p1x, p2x, g1mask, g2mask, p1mask,p2mask, r, pr)
		    #print costpg
		    #cost1 = self.train_function1(g1x, g2x, p1x, g1mask, g2mask, p1mask,r)
		    text_file.write("Epoch %s   Cost %f  Words Cost  %f \n"% (eidx+1, cost, cost1 ) )
		    #print 'Epoch ', (eidx+1), 'Cost ', cost , 'Pure Cost ', cost1
                    t2 = time.time()
                    #print "cost time: "+str(t2-t1)
		
	                        
                    if np.isnan(cost) or np.isinf(cost):
                        text_file.write( 'NaN detected\n')
		   
	        COPA, accurancy1, accruancy2, threshold, testacc = self.evaluate()
		if COPA > result0:
		    result0 = COPA
		    recordtime = eidx

		if accruancy2 > result1:
                    result1 = accruancy2
		    conceptime = eidx
		    #print COPA
		text_file.write( "\nCOPA Result: %f dev1 %f  dev2 %f  threshold %f testacc %f \n\n" % (COPA, accurancy1, accruancy2, threshold, testacc))
		text_file.write( ' ')
             

                #if(params.save):
                #    counter += 1
                #    self.saveParams(params.outfile+str(counter)+'.pickle')

                #evaluate_all(self,self.words)

            	text_file.write( 'Seen %d samples \n' % n_samples)


        except KeyboardInterrupt:
            text_file.write( "Training interupted \n")

        end_time = time.time()
	text_file.write( '\n\n\n Best COPA  Result %f at %d  Conceptnet %f at trainig time %d \n' % (result0, recordtime, result1, conceptime))
        print "total time:", (end_time - start_time)
	text_file.close()
