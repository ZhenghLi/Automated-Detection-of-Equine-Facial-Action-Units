# statistics
def statistics(pred, y, thresh=0.5):
    batch_size = pred.size(0)

    pred = pred > thresh
    pred = pred.long()

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(batch_size):
        if pred[i] == 1:
            if y[i] == 1:
                TP += 1
            elif y[i] == 0:
                FP += 1
            else:
                assert False
        elif pred[i] == 0:
            if y[i] == 1:
                FN += 1
            elif y[i] == 0:
                TN += 1
            else:
                assert False
        else:
            assert False

    statistics_list = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return statistics_list

def calc_statistics(statistics_list):
    TP = statistics_list['TP']
    FP = statistics_list['FP']
    TN = statistics_list['TN']
    FN = statistics_list['FN']

    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-20)
    precision = TP / (TP + FP + 1e-20)
    recall = TP / (TP + FN + 1e-20)
    f1_score = 2 * precision * recall / (precision + recall + 1e-20)

    return f1_score, accuracy, precision, recall

def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    else:    
        old_list['TP'] += new_list['TP']
        old_list['FP'] += new_list['FP']
        old_list['TN'] += new_list['TN']
        old_list['FN'] += new_list['FN']

    return old_list