import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, label_binarize, LabelEncoder


def eval():

    x = pd.read_csv("datasets/predictions_dataset.csv")
    real = x.REAL

    pred = x.PREDICTED
    scores = x.SCORES
    for i in range(len(scores)):
        scores[i] = scores[i][1:-1]
        scores[i] = scores[i].split()
        for j in range(len(scores[i])):
            scores[i][j] = float(scores[i][j])

    for i in range(len(scores)):
        for j in range(len(scores[0])):
            scores[i][j] = (scores[i][j], j)

    #RANK
    ranks=len(scores[0])
    CMS = dict()
    c=0
    for k in range(ranks):
        CMS[k+1]=c
        for i in range(len(real)):
            s_scores = sorted(scores[i], reverse=True)
            if s_scores[k][1]==real[i]:
                CMS[k+1]+=1
                c+=1
        CMS[k+1]=CMS[k+1]/len(real)

    prob=[0]+(list(CMS.values()))

    plt.figure()
    plt.plot(list(range(ranks+1)), prob)
    #plt.xlim([0.0, 5.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ranks')
    plt.ylabel('Prop. of identification')
    plt.title('Comulative Match Characteristic')
    plt.savefig('plot/CMC/CMC.svg', dpi=1200)
    plt.clf()

    plt.figure()
    plt.plot(list(range(5 + 1)), prob[0:6])
    # plt.xlim([0.0, 5.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ranks')
    plt.ylabel('Prop. of identification')
    plt.title('Comulative Match Characteristic at rank 5')
    plt.savefig('plot/CMC/CMC_at_rank_5.svg', dpi=1200)
    plt.clf()

    print("Score at rank 1: ", CMS[1])











