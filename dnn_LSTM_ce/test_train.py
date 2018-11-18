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

#pretrained ebmedding size
params.embedsize = int(sys.argv[1])
#regularization parameters
params.LC = float(sys.argv[2])
#mini batch size
params.batchsize = int(sys.argv[3])
#learning rate
params.eta = float(sys.argv[4])
#relation initiallization value
params.initiallization=float(sys.argv[5])
#activation function for the first layer
params.activation=str(sys.argv[6])
#activation fucntion for the second layer
params.activation2=str(sys.argv[7])
#relation embedding size
params.relsize = int(sys.argv[8])
#use how many percentage of the training data, if it's 1, then it's the full data set.
params.frac = 1
#model output file, default save it to models directory
params.outfile = 'LSTM-Softmax'+'dSize'+str(sys.argv[1])
#training data file
params.dataf = '../commonsendata/Training/new_omcs100.txt'
#way to select negative example during training, possible choice: MAX, MIX, RAND
params.type = "MAX"
params.save = False
params.constraints = False
params.evaType = 'cause'
params.usepeep = True
params.margin = 1
layersize = 100


# (words, We) = getWordmap('../commonsendata/embeddings/tuples/embeddings.skip.newtask.en.d'+str(sys.argv[1])+'.m1.w5.s0.it20.txt')
# if downloading data from http://ttic.uchicago.edu/~kgimpel/commonsense.html
(words, We) = getWordmap('../commonsendata/embeddings/embeddings.txt')
rel = getRelation('../commonsendata/Training/rel.txt')
params.outfile = "../models/"+params.outfile+".Frac:"+str(params.frac)+".Act:"+str(params.activation)+str(params.activation2)+".Batch:"+str(params.batchsize)+".LC"+str(params.LC)+".eta"+str(params.eta)+"relSize"+str(params.relsize)+"."+time.strftime("%Y%m%d-%H%M%S")+".txt"
                                #examples are shuffled data
fin=open(params.outfile,"w",0)

#Examples are shuffled data
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


tm = theano_word_model(We, words, layersize, params.embedsize, rel, params.batchsize, Rel_init, params.LC, params.eta, params.margin, params.usepeep,fin,params.initiallization,params.relsize,params.activation)
tm.train( params.data, params, We,fin)
fin.close()


