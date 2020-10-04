from sklearn.metrics import confusion_matrix

def sign(x):
    if x==0:
        return 0
    else:
        return x/abs(x)

def print_confusion_matrix(true, preds):
    return confusion_matrix(true, preds, normalize=None)


