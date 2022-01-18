import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#True and scores
#true = y_test
#scores = y_pred

#True and scores
true = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])

#ROC: False positive rate, true positive rate
fpr_roc, tpr_roc, thresholds_roc = metrics.roc_curve(true, scores, pos_label=2)
#plt.plot(fpr_roc, tpr_roc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()

#DET: False positive rate, false negative rate
fpr_det, fnr_det, thresholds_det = metrics.det_curve(true, scores, pos_label=2)
#plt.plot(fpr_det, fnr_det)
plt.xlabel('False positive rate')
plt.ylabel('False negative rate')
plt.show()

#ROC: Equal error rate
fnr = 1 - tpr_roc
eer_threshold = thresholds_roc[np.nanargmin(np.absolute((fnr - fpr_roc)))]
EER1 = fpr_roc[np.nanargmin(np.absolute((fnr - fpr_roc)))]
EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr_roc)))]

#CMC
predictions = np.random.randint(10, size=(100, 20))
labels = np.random.randint(10, size=100)

ranks = np.zeros(len(labels))
for i in range(len(labels)) :
    if labels[i] in predictions[i] :
        firstOccurance = np.argmax(predictions[i]== labels[i])
        for j in range(firstOccurance, len(labels)) :
            ranks[j] +=1


cmcScores = [float(i)/float(len(labels)) for i in ranks]
#plt.plot(ranks, cmcScores)
plt.xlabel('Ranks')
plt.ylabel('cmcScores')
plt.show()
