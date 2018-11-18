import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData
from utils import getRelation
#from adagrad import adagrad
import random
import numpy as np
#from maxpool import theano_word_model
from finalsigmoidrandom import theano_word_model
random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

#trainSize = [10,20,50,100]
def LModel(eta,batchsize,dSize,relSize, updatewords):
	trainSize = [50]

	acti = ['relu','tanh']
	evaT = ['sum','max','cause']

	layersize =dSize

	params.frac = 1.0
	params.outfile = 'New_Model_F_sigrandom'+'_eta_'+str(eta)+'_dSize_'+ str(dSize) + '_batchsize_'+ str(batchsize) + '_relSize_'+ str(relSize) + '_trainSize_'+str(trainSize[0]) + '_updatewords_' + str(updatewords)
#params.dataf = '../data/conceptnet/AddModelData/omcs_train_new'+str(trainSize[0])+'.txt'
	#params.dataf = '../data/conceptnet/AddModelData/causes_omcs.txt'
	params.dataf = '../data/conceptnet/AddModelData/new_omcs100.txt'
	params.batchsize = batchsize
	params.hiddensize = 25
	params.type = "MAX"
	params.save = True
	params.constraints = False
	params.embedsize = dSize
	params.relsize = relSize
	params.activation = acti[0]
	params.evaType = evaT[0]
	params.usepeep = True
	params.LC = 0.00001
	params.Lw = 0.01
	params.eta = eta
	params.margin = 1
	params.save= False

	(words, We) = getWordmap('../data/conceptnet/embeddings/embeddings.skip.newtask.en.d'+str(dSize)+'.m1.w5.s0.it20.txt')
	#print We.shape
	rel = getRelation('../data/conceptnet/rel.txt')
	params.outfile = "../models/"+params.outfile+"_"+str(params.LC)+"_"+str(params.Lw)+".txt"
                                #examples are shuffled data
	examples = getData(params.dataf)

	params.data = examples[0:int(params.frac*len(examples))]

	#print "Using Training Data"+params.dataf
	#print "Using Word Embeddings with Dimension "+str(dSize[0])

	#print "Training on "+str(len(params.data))
	#print "Saving models to: "+params.outfile

	Rel_init = np.zeros((35,params.relsize,params.relsize))
	for k in range(35):
		for i in range(params.relsize):
         		for j in range(params.relsize):
                  		if(i==j):
                          		#Rel_init[k][i][j] = 1+random.uniform(-0.2,0.2)
					Rel_init[k][i][j] = random.uniform(-0.2,0.2)
                  		else:
                          		Rel_init[k][i][j] = random.uniform(-0.2,0.2)

	tm = theano_word_model(We, words, layersize, params.embedsize, rel, params.relsize, Rel_init, params.LC, params.Lw, params.eta, params.margin, params.usepeep, updatewords)
	tm.train( params.data, params)

if __name__ == "__main__":
	LModel(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
	
