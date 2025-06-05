import numpy as np

def evaluate(Y_true, Y_pred, num_classes=3):
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(Y_true, Y_pred):
        confusion[true][pred] += 1
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for c in range(num_classes):
        TP = confusion[c, c]
        FP = np.sum(confusion[:, c]) - TP
        FN = np.sum(confusion[c, :]) - TP
        precision[c] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[c] = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])

    accuracy = np.trace(confusion) / np.sum(confusion)
    return accuracy, precision, recall, f1, confusion
