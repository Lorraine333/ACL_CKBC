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

def evaluate_lstm(tm,We,words,Rel,rel,relSize):

    # WSC  = evaWSC('../commonsendata/Eval/WSC/train689.txt',words,We,rel,Rel,'cause',tm,relSize)
    evaluate_conceptNet(We,words,Rel,rel,tm,relSize,'../commonsendata/Eval/conceptnet/new_omcs_dev1.txt','../commonsendata/Eval/conceptnet/new_omcs_dev2.txt','../commonsendata/Eval/conceptnet/new_omcs_test.txt')
    COPAresult = evaCOPA('../commonsendata/Eval/COPA/copa-test-keywords.xml',words,We,rel,Rel,'cause',tm,relSize)
    return COPAresult[0]


def score(gv1,gv2,words,We,rel,Rel,relSize,tm):
    score = [None, None, None]
    gv1 = gv1.reshape((relSize))
    gv2 = gv2.reshape((relSize))
    v_r = Rel[rel['causes'],:].reshape((relSize))
    
    input_vec = np.concatenate((gv1,v_r,gv2),axis = 0)
    score[2] = tm.score_func(input_vec)
    return score

def evaCOPA(evafile,words,We,rel,Rel,evaType,tm,relSize):

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
    while idx < len(q):
        qq = q[idx: idx + 100 if idx + 100 < len(q) else len(q)]
        alter1q = alter1[idx: idx + 100 if idx + 100 < len(q) else len(q)]
        alter2q = alter2[idx: idx + 100 if idx + 100 < len(q) else len(q)]
        x0, x0_mask, x0_length = tm.prepare_data(qq)
        x1, x1_mask, x1_length = tm.prepare_data(alter1q)
        x2, x2_mask, x2_length = tm.prepare_data(alter2q)
        emb0 = tm.GetVector(x0, x0_mask, x0_length)
        emb1 = tm.GetVector(x1, x1_mask, x1_length)
        emb2 = tm.GetVector(x2, x2_mask, x2_length)
        for j in range(100):
            if(causality[j]==0):
                scores1 = score(emb1[j],emb0[j],words,We,rel,Rel,relSize,tm)
                scores2 = score(emb2[j],emb0[j],words,We,rel,Rel,relSize,tm)
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
            else:
                scores1 = score(emb0[j],emb1[j],words,We,rel,Rel,relSize,tm)
                scores2 = score(emb0[j],emb2[j],words,We,rel,Rel,relSize,tm)
                if(evaType.lower()=='max'):
                    score1 = scores1[0]
                    score2 = scores2[0]
                if(evaType.lower()=='sum'):
                    score1 = scores1[1]
                    score2 = scores2[1]
                if(evaType.lower()=='cause'):
                    score2 = scores1[2]
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

def evaWSC(evafile,words,We,rel,Rel,evaType,Weight,Offset,relSize,activation):
    f = open(evafile,'r')
    lines = f.readlines()
    trueAns = []
    q = []
    Alt = []
    alter1 = []
    alter2 = []
    same = 0 
    diff = 0
    for i in xrange(0,len(lines),5):
        singleAns = lines[i+3].strip()
        trueAns.append(singleAns)
        
        parts = lines[i].strip().split('because')
        parts[0] = parts[0].strip()
        q.append(parts[0])
        
        pronoun = lines[i+1].strip()
        alt = lines[i+2].strip().split(',')
        Alt.append(alt)
        singleAlter1 = parts[1].replace(pronoun,alt[0])
        singleAlter2 = parts[1].replace(pronoun,alt[1])
        singleAlter1 = singleAlter1.replace('.','')
        singleAlter2 = singleAlter2.replace('.','')
        singleAlter1 = singleAlter1.strip()
        singleAlter2 = singleAlter2.strip()
        alter1.append(singleAlter1)
        alter2.append(singleAlter2)

    for j in xrange(len(q)):
        scores1 = score(alter1[j],q[j],words,We,rel,Rel,evaType,Weight,Offset,relSize,activation,tm)
        scores2 = score(alter2[j],q[j],words,We,rel,Rel,evaType,Weight,Offset,relSize,activation,tm)
        if(evaType.lower()=='max'):
            score1 = scores1[0]
            score2 = scores2[0]
        if(evaType.lower()=='sum'):
            score1 = scores1[1]
            score2 = scores2[1]
        if(evaType.lower()=='cause'):
            score2 = scores1[2]
            score2 = scores2[2]
   
        if(score1>score2):
            ans = Alt[j][0]
        else:
            ans = Alt[j][1]
        if(ans == trueAns[j]):
            same = same+1
        else:
            diff = diff+1
    return same/(same+diff)

def evaluate_conceptNet(We,words,Rel,rel,tm,relSize,dev1_file,dev2_file,test_file):
    threshold = get_threshold(We,words,Rel,rel,tm,relSize,dev1_file)
    dev2_accu = get_accu(We,words,Rel,rel,tm,relSize,threshold,dev2_file)
    test_accu = get_accu(We,words,Rel,rel,tm,relSize,threshold,test_file)

    print 'Dev2-Accurancy', dev2_accu
    print 'Test-Accurancy', test_accu
    print 'Threshold',threshold

def get_accu(We,words,Rel,rel,tm,relSize,threshold,filename):
    f = open(filename,'r')
    lines = f.readlines()
    lines.append('ReceivesAction\thockey\tplay on ice\t1')
    lines.append('AtLocation\trestroom\trest area\t1')
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

    x1, x1_mask, x1_length = tm.prepare_data(t1batch)
    x2, x2_mask, x2_length = tm.prepare_data(t2batch)

    v1 = tm.GetVector(x1, x1_mask, x1_length)
    v2 = tm.GetVector(x2, x2_mask, x2_length)

    for j in range(len(lines)):
        v_r = Rel[rel[r_list[j].lower()],:].reshape((relSize))
        input_vec=np.concatenate((v1[j],v_r,v2[j]),axis=0)
        softmaxScore=tm.score_func(input_vec)
        # if j == len(lines)-1:
            # print lines[j],softmaxScore
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
           
    return accurancy

def get_threshold(We,words,Rel,rel,tm,relSize,dev1_file):
    f = open(dev1_file,'r')
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

    x1, x1_mask, x1_length = tm.prepare_data(t1batch)
    x2, x2_mask, x2_length = tm.prepare_data(t2batch)

    v1 = tm.GetVector(x1, x1_mask, x1_length)
    v2 = tm.GetVector(x2, x2_mask, x2_length)

    for j in range(len(lines)):

        v_r = Rel[rel[r_list[j].lower()],:].reshape((relSize))
        input_vec=np.concatenate((v1[j],v_r,v2[j]),axis=0)
        softmaxScore=tm.score_func(input_vec)
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
    return threshold

# if __name__ == "__main__":
#     (words, We) = getWordmap('../data/conceptnet/embeddings.txt')
#     tm = theano_word_model(We)
#     rel = getRelation('../data/conceptnet/rel.txt')
#     Rel = tm.getRel()
#     evaluate_adagrad(We,words,Rel,rel)

