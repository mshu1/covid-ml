import os
import numpy as np
import pandas as pd
from pycox import models
from pycox.evaluation import EvalSurv
from pycox.datasets import metabric



def compute_cum_baseline_hazard(T, E, O):
  ind = np.argsort(-T)
  T, E, O = T[ind], E[ind], O[ind]
  baseline_hazard_dict = {}
  Oexp = np.exp(O)
  Oc = np.cumsum(Oexp, axis=0)
  groups = np.unique(T)
  baseline_cum_hazard = np.zeros((len(groups),2))
  for i,num in enumerate(groups):
    Egroup = E[T==num]
    m = np.sum(Egroup)
    tie_risk = np.sum(Egroup * O[T==num].squeeze())
    tie_hazard = np.sum(Egroup * Oexp[T==num].squeeze())
    cum_hazard = Oc[T==num][-1] #the very last harzard is desired due to ordering
    cum_hazard_array = np.ones(m) * cum_hazard
    tie_harzard_array = (np.arange(0,m)/m) * tie_hazard
    baseline_hazard = np.sum(1/(cum_hazard_array - tie_harzard_array))
    baseline_hazard_dict[num] = baseline_hazard
    baseline_cum_hazard[i,:] = np.array([num, baseline_hazard])
  base_cum = np.cumsum(baseline_cum_hazard[:,1])
  baseline_cum_hazard[:,1] = base_cum
  return baseline_cum_hazard

def predict_hazard(O, baseline_cum_hazard):
  base_cum_hazard_value =  baseline_cum_hazard[:,1]
  return np.exp(-np.exp(O) * base_cum_hazard_value)

def predict_results(base_dict, Teval, Eeval, Oeval):
  predict_haz = predict_hazard(Oeval, base_dict)
  surv = pd.DataFrame(data=predict_haz.T, index=list(base_dict[:,0]))
  ev = EvalSurv(surv, Teval, Eeval, censor_surv='km')
  return ev