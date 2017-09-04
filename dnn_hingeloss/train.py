import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData
from utils import getRelation
#from adagrad import adagrad
import random
import numpy as np
from random import choice
from random import randint
from commonsense import theano_word_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

#dsize, relsize, acti, l2reg(0.0001), relReg(0.005), batchsize(400), learningrate, lstmsize(100)
#pretrained embedding size
params.embedsize = int(sys.argv[1])
#relation embedding size
params.relsize = int(sys.argv[2])
#activation for first layer, possible inputs: none, sigmoid, rectify
params.activation = str(sys.argv[3])
#regularization parameters
params.LC = float(sys.argv[4])
#regulatrizaiton parameter for relation matrix
params.Lw = float(sys.argv[5])
#mini batch size
params.batchsize = int(sys.argv[6])
#learning rate
params.eta = float(sys.argv[7])
#size of hidden layer
params.hiddensize = int(sys.argv[8])
#way to sample negative examples. possible inputs: 'MIX','RAND','MAX'
params.type = str(sys.argv[9])
#use how many percentage of the training data, if it's 1, then it's the full data set.
params.frac = float(sys.argv[10])
params.outfile = 'DNN_Hinge'+'trainSize100dSize'+str(sys.argv[1])+'relSize'+str(sys.argv[2])+'acti'+str(sys.argv[3])
params.dataf = '../commonsendata/Training/new_omcs100.txt'
params.save = False
params.constraints = False
params.evaType = 'cause'
params.margin = 1

# (words, We) = getWordmap('../commonsendata/embeddings/tuples/embeddings.skip.newtask.en.d'+str(sys.argv[1])+'.m1.w5.s0.it20.txt')
# print We.shape
# if downloading data from http://ttic.uchicago.edu/~kgimpel/commonsense.html
(words, We) = getWordmap('../commonsendata/embeddings/embeddings.txt')
rel = getRelation('../commonsendata/Training/rel.txt')
params.outfile = "../models/"+params.outfile+"."+str(params.lam)+"."+str(params.batchsize)+"."+params.type+"."+params.activation+".txt"

examples = getData(params.dataf)

params.data = examples[0:int(params.frac*len(examples))]

print "Using Training Data"+params.dataf
print "Using Word Embeddings with Dimension "+str(sys.argv[1])

print "Training on "+str(len(params.data))+" examples using lambda="+str(params.lam)
print "Saving models to: "+params.outfile

Rel_init = np.zeros((35,params.relsize))
for m in range(35):
	for n in range(params.relsize):
		Rel_init[m][n] = random.uniform(-0.02,0.02)

tm = theano_word_model(We, words, params.hiddensize, params.embedsize, rel, params.relsize, Rel_init, params.LC, params.Lw, params.eta, params.margin, params.activation)
tm.train( params.data, params, We)

