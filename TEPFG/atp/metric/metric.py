import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
from sklearn.metrics import normalized_mutual_info_score, \
    adjusted_rand_score, rand_score

class MetricAbstract:
    def __init__(self):
        self.bigger= True
    def __str__(self):
        return self.__class__.__name__


    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")




class RMSE(MetricAbstract):
    def __init__(self):
        self.bigger= False
    def __call__(self, groundtruth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        return np.linalg.norm(pred.data - groundtruth.data)


class MAE(MetricAbstract):
    def __init__(self):
        self.bigger= False

    def __call__(self, groundtruth, pred) -> float:
        assert sp.isspmatrix_csr(pred)
        assert sp.isspmatrix_csr(groundtruth)
        return np.mean(np.abs(pred.data - groundtruth.data))


class ARI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return adjusted_rand_score(groundtruth, pred)


class RI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return rand_score(groundtruth, pred)


class NMI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return normalized_mutual_info_score(groundtruth, pred)


class ACC(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        y_true = groundtruth.astype(np.int64)
        y_pred = pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = opt.linear_sum_assignment(w.max() - w)
        total = 0
        for i in range(len(ind[0])):
            total += w[ind[0][i], ind[1][i]]
        return total * 1.0 / y_pred.size
