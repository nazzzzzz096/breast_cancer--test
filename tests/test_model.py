from model import predict_cancer
import numpy as np
from sklearn.datasets import load_breast_cancer


def test_pre_mo():
    data=load_breast_cancer()
    samp=data.data[0]
    result=predict_cancer(samp)
    assert result in [0,1]