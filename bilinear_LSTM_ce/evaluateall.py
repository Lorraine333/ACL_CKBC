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


def processJHUPPDB():
    JHUall = []
    JHUwords = []
    JHUphrases = []
    f = open('../dataset/ppdb-sample.tsv','r')
    lines = f.readlines()
    for i in lines:
        i=i.split("\t")
        score = float(i[0])
        p1 = i[2].lower()
        p2 = i[3].lower()
        if len(p1.split()) == 1 and len(p2.split()) ==1:
            #print p1, p2
            JHUwords.append((p1,p2,score))
        if len(p1.split()) > 1 and len(p2.split()) > 1:
            JHUphrases.append((p1,p2,score))
        JHUall.append((p1,p2,score))
    #print len(JHUwords)
    return (JHUall,JHUwords,JHUphrases)

def getJHUspears(data,We,words):
    gold = []
    llm_s = []
    llm_sa = []
    add_s = []
    for i in data:
        gold.append(i[2])
        llm_s.append(llm(i[0],i[1],words,We))
        llm_sa.append(absllm(i[0],i[1],words,We))
        add_s.append(add(i[0],i[1],words,We))
    return (spearmanr(gold,llm_s)[0], spearmanr(gold,add_s)[0], spearmanr(gold,llm_sa)[0])

def evaluateJHUPPDB(We,words):
    (JHUall,JHUwords,JHUphrases) = processJHUPPDB()
    jall = getJHUspears(JHUall,We,words)
    jwords = getJHUspears(JHUwords,We,words)
    jphrases = getJHUspears(JHUphrases,We,words)
    return (jall,jwords,jphrases)

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    if xdiff2 == 0 or ydiff2 == 0:
        return 0.0

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def EvalSingleSystem(testlabelfile, sysscores):

    # read in golden labels
    goldlabels = []
    goldscores = []

    hasscore = False
    with open(testlabelfile) as tf:
        for tline in tf:
            tline = tline.strip()
            tcols = tline.split('\t')
            if len(tcols) == 2:
                goldscores.append(float(tcols[1]))
                if tcols[0] == "true":
                    goldlabels.append(True)
                elif tcols[0] == "false":
                    goldlabels.append(False)
                else:
                    goldlabels.append(None)

    tp = 0
    fn = 0
    # evaluation metrics
    for i in range(len(goldlabels)):

        if goldlabels[i] == True:
            tp += 1

    # system degreed scores vs golden binary labels
    # maxF1 / Precision / Recall

    maxF1 = 0
    P_maxF1 = 0
    R_maxF1 = 0

    # rank system outputs according to the probabilities predicted
    sortedindex = sorted(range(len(sysscores)), key = sysscores.__getitem__)
    sortedindex.reverse()

    truepos  = 0
    falsepos = 0

    for sortedi in sortedindex:
        if goldlabels[sortedi] == True:
            truepos += 1
        elif goldlabels[sortedi] == False:
            falsepos += 1

        precision = 0

        if truepos + falsepos > 0:
            precision = float(truepos) / (truepos + falsepos)

        recall = float(truepos) / (tp + fn)
        f1 = 0

        #print truepos, falsepos, precision, recall

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > maxF1:
                maxF1 = f1
                P_maxF1 = precision
                R_maxF1 = recall

    # system degreed scores  vs golden degreed scores
    # Pearson correlation
    #print len(sysscores), len(goldscores)
    pcorrelation = pearson(sysscores, goldscores)

    return (pcorrelation, maxF1)

def evaluateTwitter(We,words):
    file = open('../dataset/test.data','r')
    lines = file.readlines()
    llm_scores = []
    allm_scores = []
    add_scores = []
    for i in lines:
        i =i.split("\t")
        s1 = i[2].lower()
        s2 = i[3].lower()
        llm_scores.append(llm(s1,s2,words,We))
        allm_scores.append(absllm(s1,s2,words,We))
        add_scores.append(add(s1,s2,words,We))
    (llm_c, llm_f) = EvalSingleSystem("../dataset/test.label",llm_scores)
    (add_c, add_f) = EvalSingleSystem("../dataset/test.label",add_scores)
    (allm_c, allm_f) = EvalSingleSystem("../dataset/test.label",allm_scores)
    return (llm_c, add_c, allm_c, llm_f, add_f, allm_f)

def getwordsim(file):
    file = open(file,'r')
    lines = file.readlines()
    lines.pop(0)
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            ex = (i[0],i[1],float(i[2]))
            examples.append(ex)
    #print examples
    return examples

def getsimlex(file):
    file = open(file,'r')
    lines = file.readlines()
    lines.pop(0)
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            ex = (i[0],i[1],float(i[3]))
            examples.append(ex)
    return examples

def getCorr(examples, We, words):
    gold = []
    pred = []
    num_unks = 0
    for i in examples:
        (v1,t1) = lookup_with_unk(We,words,i[0])
        (v2,t2) = lookup_with_unk(We,words,i[1])
        #print v1,v2
        pred.append(-1*cosine(v1,v2)+1)
        if t1:
            num_unks += 1
            #print i[0]
        if t2:
            num_unks += 1
        gold.append(i[2])
    return (spearmanr(pred,gold)[0], num_unks)

def evaluateWordSim(We, words):
    ws353ex = getwordsim('../dataset/wordsim353.txt')
    ws353sim = getwordsim('../dataset/wordsim-sim.txt')
    ws353rel = getwordsim('../dataset/wordsim-rel.txt')
    simlex = getsimlex('../dataset/SimLex-999.txt')
    (c1,u1) = getCorr(ws353ex,We,words)
    (c2,u2) = getCorr(ws353sim,We,words)
    (c3,u3) = getCorr(ws353rel,We,words)
    (c4,u4) = getCorr(simlex,We,words)
    return ([c1,c2,c3,c4],[u1,u2,u3,u4])

def llm(p1,p2,words,We):
    p1 = p1.split()
    p2 = p2.split()
    total = 0
    for i in p1:
        v1 = lookup(We,words,i)
        max = -5
        for j in p2:
            v2 = lookup(We,words,j)
            score = -1*cosine(v1,v2)+1
            if(score > max):
                max = score
        total += max
    llm_score = 0.5*total / len(p1)
    total = 0
    for i in p2:
        v1 = lookup(We,words,i)
        max = -5
        for j in p1:
            v2 = lookup(We,words,j)
            score = -1*cosine(v1,v2)+1
            if(score > max):
                max = score
        total += max
    llm_score += 0.5*total / len(p2)
    return llm_score

def absllm(p1,p2,words,We):
    p1 = p1.split()
    p2 = p2.split()
    total = 0
    for i in p1:
        v1 = lookup(We,words,i)
        max = 0
        for j in p2:
            v2 = lookup(We,words,j)
            score = -1*cosine(v1,v2)+1
            if(abs(score) > abs(max)):
                max = score
        total += max
    llm_score = 0.5*total / len(p1)
    total = 0
    for i in p2:
        v1 = lookup(We,words,i)
        max = 0
        for j in p1:
            v2 = lookup(We,words,j)
            score = -1*cosine(v1,v2)+1
            if(abs(score) > abs(max)):
                max = score
        total += max
    llm_score += 0.5*total / len(p2)
    return llm_score

def add(p1,p2,words,We):
    p1 = p1.split()
    p2 = p2.split()
    accumulator = np.zeros(lookup(We,words,p1[0]).shape)
    for i in p1:
        v = lookup(We,words,i)
        accumulator = accumulator + v
    p1_emb = accumulator / len(p1)
    accumulator = np.zeros(lookup(We,words,p2[0]).shape)
    for i in p2:
        v = lookup(We,words,i)
        accumulator = accumulator + v
    p2_emb = accumulator / len(p1)
    return -1*cosine(p1_emb,p2_emb)+1


def scoreannoppdb(f,We,words):
    f = open(f,'r')
    lines = f.readlines()
    allm_preds = []
    llm_preds = []
    add_preds = []
    gold = []
    for i in lines:
        i=i.strip()
        i=i.split('|||')
        (p1,p2,score) = (i[0].strip(),i[1].strip(),float(i[2]))
        llm_preds.append(llm(p1,p2,words,We))
        allm_preds.append(absllm(p1,p2,words,We))
        add_preds.append(add(p1,p2,words,We))
        gold.append(score)
    return (spearmanr(llm_preds,gold)[0], spearmanr(add_preds,gold)[0], spearmanr(allm_preds,gold)[0])

def scoreSE(f,We,words):
    f = open(f,'r')
    lines = f.readlines()
    lines.pop(0)
    examples=[]
    llm_preds = []
    add_preds = []
    allm_preds = []
    gold = []
    for i in lines:
        i=i.strip()
        i=i.split('\t')
        (p1,p2,score) = (i[1].strip(),i[2].strip(),float(i[3]))
        llm_preds.append(llm(p1,p2,words,We))
        add_preds.append(add(p1,p2,words,We))
        allm_preds.append(absllm(p1,p2,words,We))
        gold.append(score)
    return (spearmanr(llm_preds,gold)[0], spearmanr(add_preds,gold)[0], spearmanr(allm_preds,gold)[0])

def evaluateAnno(We, words):
    return scoreannoppdb('../dataset/ppdb_test.txt',We,words)

def evaluateSE(We, words):
    return scoreSE('../dataset/SICK_trial.txt',We,words)

def evaluate_adagrad(We,words,Rel,rel,Weight,Offset,relSize,activation):
    spearman = evaluate_conceptNet(We,words,Rel,rel,Weight,Offset,relSize,activation)
    COPAresult = evaCOPA('../data/conceptnet/COPA/copa-dev-key.xml',words,We,rel,Rel,'max',Weight,Offset,relSize,activation)
    COPA = COPAresult[0]
    WSC = evaWSC('../data/conceptnet/WSC/our_train.txt',words,We,rel,Rel,'max',Weight,Offset,relSize,activation)
    (corr, unk) = evaluateWordSim(We,words)
    #print 'Word Sim Result: ',corr,unk
    (llm_s, add_s, allm_s) = evaluateAnno(We,words)
    (llm_se, add_se, allm_se) = evaluateSE(We,words)
#    ((j1,j2,j3),(j4,j5,j6),(j7,j8,j9)) = evaluateJHUPPDB(We,words)
#    (llm_tw, add_tw, allm_tw, _, _, _) = evaluateTwitter(We,words)
    s = "wsim: {0:.4f} {1:.4f} {2:.4f} {3:.4f} "
    s += "ppdb: {4:.4f} {5:.4f} {6:.4f} "
    s += "semeval: {7:.4f} {8:.4f} {9:.4f} "
#    s += "JHU: all {10:.4f} {11:.4f} {12:.4f} | word {13:.4f} | phrase {14:.4f} {15:.4f} {16:.4f} "
#    s += "Twitter: {17:.4f} {18:.4f} {19:.4f}"
#    s=s.format(corr[0], corr[1], corr[2], corr[3], llm_s, add_s, allm_s, llm_se, add_se, allm_se, j1, j2, j3, j4, j7, j8, j9, llm_tw, add_tw, allm_tw)
    s=s.format(corr[0], corr[1], corr[2], corr[3], llm_s, add_s, allm_s, llm_se, add_se, allm_se)
    print s
    print 'Spearmanr Result: ',spearman
    print 'COPA Result: ', COPA
    print 'WSC Result: ', WSC
    return spearman,COPAresult[1],COPAresult[2],COPAresult[3]

def evaluate_lstm(tm,We,words,Rel,rel,relSize):
   
    #spearman = evaluate_conceptNet(We,words,Rel,rel,Weight,Offset,relSize,activation)
    #COPAresult = evaCOPA('../data/conceptnet/COPA/copa-dev-key.xml',words,We,rel,Rel,'cause',tm,relSize)
    COPAresult = evaCOPA('../data/conceptnet/COPA/copa-dev.xml',words,We,rel,Rel,'cause',tm,relSize)
    accurancy1, accruancy2, threshold, test =evaluate_conceptNet(words, We, rel, Rel, tm, relSize)
    return COPAresult[0], accurancy1, accruancy2, threshold, test


def score(gv1,gv2,words,We,rel,Rel,relSize):
    size = relSize
    result = {}
    scores = []

    for k,v in rel.items():
        relMatrix = Rel[rel[k]*size:rel[k]*size+size,:]
        temp1 = np.dot(gv1, relMatrix)
        result[k] = np.inner(temp1,gv2)
    #print result

    
    result1 = sorted(result.items(), key=lambda x: x[1], reverse = True)
    scores.append(result1[1][1])
    total = 0
    for i in result1:
        total = total + i[1]
    scores.append(total)
    scores.append(result.get('causes'))

    return scores

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
	causalityq = causality[idx: idx + 100 if idx + 100 < len(q) else len(q)]
        x0, x0_mask = tm.prepare_data(qq)
	x1, x1_mask = tm.prepare_data(alter1q)
	x2, x2_mask = tm.prepare_data(alter2q)
	emb0 = tm.GetVector(x0, x0_mask)
	emb1 = tm.GetVector(x1, x1_mask)
	emb2 = tm.GetVector(x2, x2_mask)
	for j in range(100):
		
        	if(causalityq[j]==0):
    	    		scores1 = score(emb1[j],emb0[j],words,We,rel,Rel,relSize)
    	    		scores2 = score(emb2[j],emb0[j],words,We,rel,Rel,relSize)
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
            		scores1 = score(emb0[j],emb1[j],words,We,rel,Rel,relSize)
            		scores2 = score(emb0[j],emb2[j],words,We,rel,Rel,relSize)
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
    # print 'totalScore1',len(totalScore1)
    # print 'totalScore2',len(totalScore2)
    # print 'trueAns',len(trueAns)
    #print same, diff	
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
	scores1 = score(alter1[j],q[j],words,We,rel,Rel,evaType,Weight,Offset,relSize,activation)
        scores2 = score(alter2[j],q[j],words,We,rel,Rel,evaType,Weight,Offset,relSize,activation)
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

def evaluate_conceptNet(words, We, rel, Rel, tm, relSize):
    threshold, accurancy1 = get_threshold('../data/conceptnet/new_omcs_dev1.txt', We, words, rel, Rel, tm, relSize)
    f = open('../data/conceptnet/new_omcs_dev2.txt','r')
    lines = f.readlines()
    S = []
    T = []
    T1 = []
    T2 = []
    R = []
    Exp_S = []
    result = []
    for i in lines:
        i = i.strip()
        i = i.split('\t')
        (r,t1,t2,score) = (i[0].strip(),i[1].strip(),i[2].strip(),float(i[3]))
	temp1 = lookupwordID(We,words,t1)
        temp2 = lookupwordID(We,words,t2)
	T1.append(temp1)
	T2.append(temp2)
	tp = rel[r.lower()]
	c = Rel[tp*relSize: tp*relSize + relSize,:]
	R.append(c)

    x1, x1_mask = tm.prepare_data(T1)
    x2, x2_mask = tm.prepare_data(T2)     
    emb1 = tm.GetVector(x1, x1_mask)
    emb2 = tm.GetVector(x2, x2_mask)
    for j in range(len(R)):

        gv1 = emb1[j]
        gv2 = emb2[j]
        v_r = R[j]

        temp1 = np.dot(gv1, v_r)
        exp_score = np.inner(temp1,gv2)
        Exp_S.append(exp_score)

    right = 0
    wrong = 0
    accurancy2 = 0
    for j1 in xrange(int(len(Exp_S)/2)):
        if(Exp_S[j1]>=threshold):
            right = right+1
        else:
            wrong = wrong+1
    for j2 in xrange(int(len(Exp_S)/2),int(len(Exp_S)),1):
        if(Exp_S[j2]<=threshold):
            right = right+1
        else:
            wrong = wrong+1
    accurancy2 = (right/(len(Exp_S)))
    
    f = open('../data/conceptnet/new_omcs_test.txt','r')
    lines = f.readlines()
    S = []
    T = []
    T1 = []
    T2 = []
    R = []
    Exp_S = []
    result = []
    for i in lines:
        i = i.strip()
        i = i.split('\t')
        (r,t1,t2,score) = (i[0].strip(),i[1].strip(),i[2].strip(),float(i[3]))
        temp1 = lookupwordID(We,words,t1)
        temp2 = lookupwordID(We,words,t2)
        T1.append(temp1)
        T2.append(temp2)
        tp = rel[r.lower()]
        c = Rel[tp*relSize: tp*relSize + relSize,:]
        R.append(c)

    x1, x1_mask = tm.prepare_data(T1)
    x2, x2_mask = tm.prepare_data(T2)
    emb1 = tm.GetVector(x1, x1_mask)
    emb2 = tm.GetVector(x2, x2_mask)
    for j in range(len(R)):

        gv1 = emb1[j]
        gv2 = emb2[j]
        v_r = R[j]

        temp1 = np.dot(gv1, v_r)
        exp_score = np.inner(temp1,gv2)
        Exp_S.append(exp_score)

    right = 0
    wrong = 0
    test = 0
    for j1 in xrange(int(len(Exp_S)/2)):
        if(Exp_S[j1]>=threshold):
            right = right+1
        else:
            wrong = wrong+1
    for j2 in xrange(int(len(Exp_S)/2),int(len(Exp_S)),1):
        if(Exp_S[j2]<=threshold):
            right = right+1
        else:
            wrong = wrong+1
    test = (right/(len(Exp_S)))
    #print 'Dev2-Accurancy',accurancy
    #print 'Threshold',threshold
    return accurancy1, accurancy2, threshold, test

def get_threshold(evafile, We, words, rel, Rel, tm, relSize):
    f1 = open(evafile,'r')
    lines = f1.readlines()
    S = []
    T = []
    T1 = []
    T2 = [] 
    R = []
    Exp_S = []
    result = []
    for i in lines:
        i = i.strip()
        i = i.split('\t')
        (r,t1,t2,score) = (i[0].strip(),i[1].strip(),i[2].strip(),float(i[3]))
	temp1 = lookupwordID(We,words,t1)
	temp2 = lookupwordID(We,words,t2)
        T1.append(temp1)
        T2.append(temp2)
        tp = rel[r.lower()]
        c = Rel[tp*relSize : tp*relSize + relSize, :]
        R.append(c)
	        
    x1, x1_mask = tm.prepare_data(T1)
    x2, x2_mask = tm.prepare_data(T2)
    emb1 = tm.GetVector(x1, x1_mask)
    emb2 = tm.GetVector(x2, x2_mask)
    for j in range(len(R)):

        gv1 = emb1[j]
        gv2 = emb2[j]
        v_r = R[j]
        temp1 = np.dot(gv1, v_r)
        exp_score = np.inner(temp1,gv2)
        Exp_S.append(exp_score)

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
            if(Exp_S[j2]<=temp_thr):
                right = right+1
            else:
                wrong = wrong+1
        if((right/(len(Exp_S)))>accurancy):
            accurancy = (1.0 *right/(len(Exp_S)))
            threshold = temp_thr
        right = 0
        wrong = 0
           
    #print 'Dev1-Accurancy',accurancy
    return threshold,accurancy

if __name__ == "__main__":
    (words, We) = getWordmap('../data/conceptnet/embeddings.txt')
    tm = theano_word_model(We)
    rel = getRelation('../data/conceptnet/rel.txt')
    Rel = tm.getRel()
    evaluate_adagrad(We,words,Rel,rel)

