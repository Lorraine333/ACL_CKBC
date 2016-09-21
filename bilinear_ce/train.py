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

#pretrained embedding size
params.embedsize = int(sys.argv[1])
#matrix relation size
params.relsize = int(sys.argv[2])
#regularization parameters
params.LC = float(sys.argv[3])
#regulatrizaiton parameter for relation matrix
params.Lw = float(sys.argv[4])
#learning rate
params.eta = float(sys.argv[5])
#mini batch size
params.batchsize = int(sys.argv[6])
#way to sample negative examples. possible input: 'MIX','RAND','MAX'
params.type = str(sys.argv[7])
#training epochs
params.epochs = int(sys.argv[8])
#use how many percentage of the training data, if it's 1, then it's the full data set.
params.frac = float(sys.argv[9])
params.outfile = 'Bilinear_ce'+'trainSize300frac'+str(params.frac)+'dSize'+str(sys.argv[1])+'relSize'+str(sys.argv[2])+'acti'+str(sys.argv[3])
#use normal training dataset 
params.dataf = '../commonsendata/Training/new_omcs100.txt'

#when double the training dataset
# params.dataf = '../commonsendata/Training/new_omcs600.txt'

#if you want to save the model, just change this to 'True'
params.save = False
params.constraints = False
params.activation = "tanh"
params.evaType = 'cause'
params.usepeep = True
params.margin = 1


(words, We) = getWordmap('../commonsendata/embeddings/tuples/embeddings.skip.newtask.en.d'+str(sys.argv[1])+'.m1.w5.s0.it20.txt')
rel = getRelation('../commonsendata/Training/rel.txt')
params.outfile = "../models/"+params.outfile+"."+str(params.lam)+"."+str(params.batchsize)+"."+params.type+"."+params.activation+".txt"

#Examples are shuffled data
examples = getData(params.dataf)

params.data = examples[0:int(params.frac*len(examples))]

print "Using Training Data"+params.dataf
print "Using Word Embeddings with Dimension "+str(sys.argv[1])

print "Training on "+str(len(params.data))+" examples using lambda="+str(params.lam)
print "Saving models to: "+params.outfile

#Initialize relation matrix
Rel_init = np.zeros((35,params.relsize,params.relsize))
for k in range(35):
	for i in range(params.relsize):
         	for j in range(params.relsize):
                  	if(i==j):
                          	Rel_init[k][i][j] = 1+random.uniform(-0.2,0.2)
                  	else:
                          	Rel_init[k][i][j] = random.uniform(-0.2,0.2)

tm = theano_word_model(We, words, params.embedsize, rel, params.relsize, Rel_init, params.LC, params.Lw, params.eta, params.margin, params.usepeep, params.activation)
tm.train( params.data, params, We)

