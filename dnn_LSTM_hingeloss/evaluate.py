from __future__ import division
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from utils import lookup_with_unk
from utils import lookup
from utils import getWordmap
from utils import getRelation
#from utils import getVec
from utils import Relu
from utils import Sigmoid
from utils import lookupwordID
import numpy as np
import math

def evaluate_lstm(tm,We,words,Rel,rel,memsize,relSize,fin):
   
    #spearman = evaluate_conceptNet(We,words,Rel,rel,Weight,Offset,relSize,activation)
    Accurancy = evaluate_conceptNet(We,words,Rel,rel,tm,memsize,relSize,fin)
    COPAresult = evaCOPA('../../commonsendata/Eval/COPA/copa-dev-keywords.xml',words,We,rel,Rel,'cause',tm,memsize,relSize,fin)
    # print 'WSC',WSC
    return COPAresult[0]
    #return Accurancy

def score(gv,words,We,rel,Rel,tm,memsize,relSize):
    score = [None, None, None]
    gv=gv.reshape((1,memsize))
    v_r = Rel[rel['causes'],:].reshape((1,relSize))
    input_vec=np.concatenate((gv,v_r),axis=1)
    score[2]=tm.GetVectorNew(input_vec)
    return score


def evaCOPA(evafile,words,We,rel,Rel,evaType,tm,memsize,relSize,fin):
    f = open(evafile,'r')
    lines = f.readlines()
    trueAns = []
    q = []
    alter1 = []
    alter2 = []
    causality = []
    same = 0 
    diff = 0
    totalScore1 = []
    totalScore2 = []
    for i in xrange(4,len(lines)-1,6):
        singleAns = lines[i][lines[i].find('alternative=')+13:lines[i].find('>')-1]
        trueAns.append(singleAns)
        
        if(lines[i].find('effect')!=-1):
            causality.append(1)
        else:
            causality.append(0)
        
        singleq = lines[i+1][lines[i+1].find('<p>')+3:lines[i+1].find('</p>')-1]
        temp0 = lookupwordID(We,words,singleq)
        q.append(temp0)
        
        singleAlter1 = lines[i+2][lines[i+2].find('<a1>')+4:lines[i+2].find('</a1>')-1]
        temp1 = lookupwordID(We, words,singleAlter1)
        alter1.append(temp1)
        
        singleAlter2 = lines[i+3][lines[i+3].find('<a2>')+4:lines[i+3].find('</a2>')-1]
        temp2 = lookupwordID(We,words,singleAlter2)
        alter2.append(temp2)

    idx=0
    delim= (lookupwordID(We, words, "#"))

    while idx < len(q):
        qq = q[idx: idx + 100 if idx + 100 < len(q) else len(q)]
        alter1q = alter1[idx: idx + 100 if idx + 100 < len(q) else len(q)]
        alter2q = alter2[idx: idx + 100 if idx + 100 < len(q) else len(q)]
        alt1=[]
        alt2=[]
        for j in range(100):
            if(causality[j]==0):
                alt1.append(alter1q[j]+delim+qq[j])
                alt2.append(alter2q[j]+delim+qq[j])
            else:
                alt1.append(qq[j]+delim+alter1q[j])
                alt2.append(qq[j]+delim+alter2q[j])

        x_alt1,x_alt1_mask=tm.prepare_data(alt1)
        x_alt2,x_alt2_mask=tm.prepare_data(alt2)

        emb_alt1=tm.GetVector(x_alt1,x_alt1_mask)
        emb_alt2=tm.GetVector(x_alt2,x_alt2_mask)

        for j in range(100):
    	    scores1 = score(emb_alt1[j],words,We,rel,Rel,tm,memsize,relSize)
    	    scores2 = score(emb_alt2[j],words,We,rel,Rel,tm,memsize,relSize)
            if(evaType.lower()=='max'):
                score1 = scores1[0]
                score2 = scores2[0]
            if(evaType.lower()=='sum'):
               	score1 = scores1[1]
                score2 = scores2[1]
            if(evaType.lower()=='cause'):
                score1 = scores1[2]
                score2 = scores2[2]
            totalScore1.append(scores1)
            totalScore2.append(scores2)
            if(score1>score2):
            	ans = 1
            else:
            	ans = 2
            if(ans == int(trueAns[idx])):
            	same = same+1
            else:
            	diff = diff+1
        idx = idx + 100
    return same/(same+diff),totalScore1,totalScore2,trueAns

def evaluate_conceptNet(We,words,Rel,rel,tm,memsize,relSize,fin):
    threshold = get_threshold(We,words,Rel,rel,tm,memsize,relSize,fin)
    f = open('../../commonsendata/Eval/conceptnet/new_omcs_dev2.txt','r')
    lines = f.readlines()
    Exp_S = []
    r_list = []
    t1_list = []
    t2_list = []
    for i in lines:
        i = i.strip()
        i = i.split('\t')
        (r,t1,t2,score) = (i[0].strip(),i[1].strip(),i[2].strip(),float(i[3]))
        t1id = lookupwordID(We,words,t1)
        t2id = lookupwordID(We,words,t2)
        t1_list.append(t1id)
        t2_list.append(t2id)
        r_list.append(r)
    
    t1batch = t1_list
    t2batch= t2_list

    delim= (lookupwordID(We, words, "#"))

    batchTuple=[a+delim+b for a,b in zip(t1batch,t2batch)]
    xx,xx_mask=tm.prepare_data(batchTuple)
    vector=tm.GetVector(xx,xx_mask)

    for j in range(len(lines)):

        v_r = Rel[rel[r_list[j].lower()],:].reshape((1,relSize))
        gvector=vector[j].reshape((1,memsize))
        input_vec=np.concatenate((gvector,v_r),axis=1)

        
        #input_vec = np.concatenate((gv1,v_r,gv2),axis = 1)
        softmaxScore=tm.GetVectorNew(input_vec)


        Exp_S.append(softmaxScore[0][0])

    right = 0
    wrong = 0
    accurancy = 0
    for j1 in xrange(int(len(Exp_S)/2)):
        if(Exp_S[j1]>=threshold):
            right = right+1
        else:
            wrong = wrong+1
    for j2 in xrange(int(len(Exp_S)/2),int(len(Exp_S)),1):
        if(Exp_S[j2]<threshold):
            right = right+1
        else:
            wrong = wrong+1
    accurancy = (right/(len(Exp_S)))
           
    print 'Dev2-Accurancy',accurancy
    fin.write('Dev2-Accurancy'+str(accurancy)+"\n")
    print 'Threshold',threshold
    fin.write('Threshold'+str(threshold)+"\n")
    return accurancy

def get_threshold(We,words,Rel,rel,tm,memsize,relSize,fin):
    #f = open('../data/conceptnet/AddModelData/omcs_dev1.txt','r')
    #f = open('../../commonsendata/Eval/conceptnet/omcs_dev1.txt','r')
    f = open('../../commonsendata/Eval/conceptnet/new_omcs_dev1.txt','r')
    lines = f.readlines()
    Exp_S = []
    r_list = []
    t1_list = []
    t2_list = []
    for i in lines:
        i = i.strip()
        i = i.split('\t')
        (r,t1,t2,score) = (i[0].strip(),i[1].strip(),i[2].strip(),float(i[3]))
        t1id = lookupwordID(We,words,t1)
        t2id = lookupwordID(We,words,t2)
        t1_list.append(t1id)
        t2_list.append(t2id)
        r_list.append(r)
    
    t1batch = t1_list[0:len(lines)]
    t2batch= t2_list[0:len(lines)]
    print 't1batch: ', len(t1batch)

    delim= (lookupwordID(We, words, "#"))


    batchTuple=[a+delim+b for a,b in zip(t1batch,t2batch)]
    xx,xx_mask=tm.prepare_data(batchTuple)
    vector=tm.GetVector(xx,xx_mask)


    for j in range(len(lines)):

        v_r = Rel[rel[r_list[j].lower()],:].reshape((1,relSize))
        vectorg=vector[j].reshape((1,memsize))
        input_vec=np.concatenate((vectorg,v_r),axis=1)

        softmaxScore=tm.GetVectorNew(input_vec)
        Exp_S.append(softmaxScore[0][0])
    right = 0
    wrong = 0
    threshold = 0
    accurancy = 0
    binaryScore = []
    Exp_S_sorted = sorted(Exp_S)
    for j in xrange(len(Exp_S)):
        temp_thr= Exp_S_sorted[j]
        for j1 in xrange(int(len(Exp_S)/2)):
            if(Exp_S[j1]>=temp_thr):
                right = right+1
            else:
                wrong = wrong+1
        for j2 in xrange(int(len(Exp_S)/2),int(len(Exp_S)),1):
            if(Exp_S[j2]<temp_thr):
                right = right+1
            else:
                wrong = wrong+1
        if((right/(len(Exp_S)))>accurancy):
            accurancy = (right/(len(Exp_S)))
            threshold = temp_thr
        right = 0
        wrong = 0
           
    print 'Dev1-Accurancy',accurancy
    fin.write('Dev1-Accurancy'+str(accurancy)+"\n")
    return threshold

if __name__ == "__main__":
    (words, We) = getWordmap('../data/conceptnet/embeddings.txt')
    tm = theano_word_model(We)
    rel = getRelation('../data/conceptnet/rel.txt')
    Rel = tm.getRel()
    evaluate_adagrad(We,words,Rel,rel)

