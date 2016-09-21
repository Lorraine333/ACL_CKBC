from scipy.io import loadmat
import numpy as np
import math
from random import shuffle
from random import choice
from random import randint
import theano
from theano import tensor as T

def lookup(We,words,w):
    if w in words:
        return We[words[w],:]
    else:
        #print 'find UUUNKKK words',w
        return We[words['UUUNKKK'],:]

def lookupIDX(We,words,w):
    if w in words:
        return words[w]
    else:
        #print 'find UUUNKKK words',w
        return words['UUUNKKK']

def lookupRelIDX(We,words,w):
    w = w.lower()
    if w in words:
        return words[w]
    else:
        #print 'find UUUNKKK words',w
        return words['UUUNKKK']

def lookup_with_unk(We,words,w):
    if w in words:
        return We[words[w],:],False
    else:
        #print 'find Unknown Words in WordSim Task',w
        return We[words['UUUNKKK'],:],True

def lookupwordID(We,words,w):
    #w = w.strip()
    result = []
    array = w.split(' ')
    for i in range(len(array)):
        if(array[i] in words):
            result.append(words[array[i]])
        else:
            #print "Find Unknown Words ",w
            result.append(words['UUUNKKK'])
    return result

def getData(f):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            e = (i[0], i[1], i[2], float(i[3]))
            examples.append(e)
    shuffle(examples)
    return examples

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    words['EXXXXAR'] = n+1
    We.append([0]*len(v))
    return (words, np.matrix(We))

def getRelation(relationfile):
    rel = {}
    f = open(relationfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i.strip()
        rel[i] = n
    return rel

#modified
def getPairMax(label,vec_r,vec,idx,d,We,words,rel,Rel,wi,wj,Weight,Offset,activation):
    min = -5000
    best = None
    for i in range(len(d)):
        if i == idx:
            continue
        (r,w1,w2,l) = d[i]
        v1 = getVec(We,words,w1)

        if(activation.lower()=='relu'):
            gv1 = Relu(np.dot(Weight,v1)+Offset[0])
            gv2 = Relu(np.dot(Weight,vec)+Offset[0])
        if(activation.lower()=='tanh'):
            gv1 = np.tanh(np.dot(Weight,v1)+Offset[0])
            gv2= np.tanh(np.dot(Weight,vec)+Offset[0])
        if(activation.lower()=='sigmoid'):
            gv1 = Sigmoid(np.dot(Weight,v1)+Offset[0])
            gv2= Sigmoid(np.dot(Weight,vec)+Offset[0])
        temp1 = np.dot(gv1, vec_r)
        np1 = np.inner(temp1,gv2)

        if(np1 > min and not(wi == w1) and not(wj==w1)):
            min = np1
            best = w1
    return best

def getPairRand(label,vec,idx,d,We,words,wi,wj):
    wpick = None
    while(wpick == None or wpick == wi or wpick == wj):
        ww = choice(d)
        ridx = randint(0,1)
        wpick = ww[ridx]
        #print wpick
    return wpick

def getPairMix(label,vec,idx,d,We,words,wi,wj):
    r1 = randint(0,1)
    if r1 == 1:
        return getPairMax(label,vec,idx,d,We,words,wi,wj,Weight,Offset,activation)
    else:
        return getPairRand(label,vec,idx,d,We,words,wi,wj)

def getVec(We,words,t):
    t = t.strip()
    array = t.split(' ')
    if array[0] in words:
        vec = We[words[array[0]],:]
    else:
        #print 'find UUUNKKK words',array[0].lower()
        vec = We[words['UUUNKKK'],:]
    for i in range(len(array)-1):
        #print array[i+1]
        if array[i+1] in words:
            vec = vec + We[words[array[i+1]],:]
        else:
            #print 'can not find corresponding vector:',array[i+1].lower()
            vec = vec + We[words['UUUNKKK'],:]
    vec = vec/len(array)
    return vec

def getPairs(d, words, We, rel, Rel, type, size,Weight,Offset,activation):
    pairs = []
    for i in range(len(d)):
        (r, t1, t2, s) = d[i]
        v1 = getVec(We,words,t1)
        v2 = getVec(We,words,t2)
        v_r = Rel[rel[r.lower()]*size:rel[r.lower()]*size+size,:]
        p1 = None
        #p2 = None
        if type == "MAX":
            #print w1
            #only change the first term
            p1 = getPairMax(s,v_r,v2,i,d,We,words,rel,Rel,t1,t2,Weight,Offset,activation)
        if type == "RAND":
            #print w1
            p1 = getPairRand(s,v1,i,d,We,words,rel,Rel,r,t1,t2)
        if type == "MIX":
            #print w1
            p1 = getPairMix(s,v1,i,d,We,words,rel,Rel,r,t1,t2)
        pairs.append(p1)
    # 'getPairs'+str(len(pairs))
    #print pairs
    return pairs

def getPairsBatch(d, words, We, rel, Rel, batchsize, type, size,Weight,Offset,activation):
    idx = 0
    pairs = []
    while idx < len(d):
        batch = d[idx: idx + batchsize if idx + batchsize < len(d) else len(d)]
        if(len(batch) <= 2):
            print "batch too small."
            continue #just move on because pairing could go faulty
        p = getPairs(batch,words,We,rel,Rel,type,size,Weight,Offset,activation)
        pairs.extend(p)
        idx += batchsize
    #print 'getPairsBatch'+str(len(pairs))
    return pairs

def convertToIndex(e, words, We, rel, Rel):
    if len(list(e))>2:
        (r,p1,p2,s) = e
        new_e = (lookupRelIDX(Rel, rel, r),lookupwordID(We, words, p1), lookupwordID(We, words, p2), float(s))
        # print new_e
        return new_e
    else:
        p1 = e[0]
        new_e = (lookupwordID(We, words, p1))
        #print new_e
        return new_e

def ReluT(x):
    return T.switch(x<0, 0 ,x)

def Relu(x):
    result = np.zeros(x.shape)
    for i in xrange(result.shape[0]):
        for j in xrange(result.shape[1]):
            if(x[i][j]>0):
                result[i][j]=x[i][j]
    return result

def Sigmoid(x):
    result = np.zeros(x.shape)
    for i in xrange(result.shape[0]):
        for j in xrange(result.shape[1]):
            result[i][j] = 1 / (1 + math.exp(-x[i][j]))
    return result


