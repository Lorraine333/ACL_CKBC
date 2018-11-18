class params(object):

    def __init__(self, lam=1E-5, lam1=1E-5 , dataf='../data/ppdb-words.txt',
        batchsize=10, margin=1, epochs=20, eta = 0.05, eta1 = 0.05, eta2 = 0.05, eta3  = 0.05, evaluate=True, save=False,
        hiddensize=25, outfile="test.out", type = "RAND", normalize = True, embedsize = 50, relsize = 25, activation = 'Relu'):

        self.lam=lam
        self.lam1 = lam1
        self.dataf = dataf
        self.batchsize = batchsize
        self.margin = margin
        self.epochs = epochs
        self.eta = eta
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.evaluate = evaluate
        self.save = save
        self.data = []
        self.hiddensize = hiddensize
        self.outfile = outfile
        self.type = type
        self.normalize = normalize
        self.embedsize = embedsize
        self.relsize = relsize 
        self.activation = activation
