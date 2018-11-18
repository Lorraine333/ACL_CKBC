import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData
from utils import getRelation
#from adagrad import adagrad
import random
import numpy as np
import time
from commonsense import theano_word_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()


params.frac = 1
params.outfile = 'LSTM-Softmax'+'dSize'+str(sys.argv[1])
params.dataf = '../../commonsendata/Training/new_omcs100.txt'
params.batchsize = int(sys.argv[3])
params.type = "MAX"
params.save = False
params.constraints = False
params.embedsize = int(sys.argv[1])
params.relsize = int(sys.argv[8])
params.evaType = 'cause'
params.usepeep = True
params.LC = float(sys.argv[2])
params.eta = float(sys.argv[4])
params.margin = 1
params.initiallization=float(sys.argv[5])
layersize = 100
params.activation=str(sys.argv[6])
params.activation2=str(sys.argv[7])

# (words, We) = getWordmap('../../commonsendata/embeddings/tuples/embeddings.skip.newtask.en.d'+str(sys.argv[1])+'.m1.w5.s0.it20.txt')
# print We.shape
# if downloading data from http://ttic.uchicago.edu/~kgimpel/commonsense.html
(words, We) = getWordmap('../commonsendata/embeddings/embeddings.txt')
rel = getRelation('../../commonsendata/Training/rel.txt')
params.outfile = "models/"+params.outfile+".Frac:"+str(params.frac)+".Act:"+str(params.activation)+str(params.activation2)+".Batch:"+str(params.batchsize)+".LC"+str(params.LC)+".eta"+str(params.eta)+"relSize"+str(params.relsize)+"."+time.strftime("%Y%m%d-%H%M%S")+".txt"
                                #examples are shuffled data
fin=open(params.outfile,"w",0)

examples = getData(params.dataf)

params.data = examples[0:int(params.frac*len(examples))]

print "Using Training Data"+params.dataf
fin.write("Using Training Data"+params.dataf+"\n")
print "Using Word Embeddings with Dimension "+str(sys.argv[1])
fin.write("Using Word Embeddings with Dimension "+str(sys.argv[1])+"\n")

print "Training on "+str(len(params.data))+" examples using lambda="+str(params.lam)
fin.write("Training on "+str(len(params.data))+" examples using lambda="+str(params.lam)+"\n")
print "Saving models to: "+params.outfile
fin.write("Saving models to: "+params.outfile+"\n")

Rel_init = np.zeros((35,params.relsize))
for m in range(35):
	for n in range(params.relsize):
		Rel_init[m][n] = random.uniform(-(params.initiallization),params.initiallization)


tm = theano_word_model(We, words, layersize, params.embedsize, rel, params.batchsize, Rel_init, params.LC, params.eta, params.margin, params.usepeep,fin,params.initiallization,params.relsize,params.activation,params.activation2)
tm.train( params.data, params, We,fin)
fin.close()


