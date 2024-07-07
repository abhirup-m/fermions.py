import scipy.linalg
import numpy as np
import base

def AddDicts(dict1, dict2):
    outputDict = dict1.copy()
    for (k,v) in dict2.items():
        if k in outputDict.keys():
            outputDict[k] += v
        else:
            outputDict[k] = v
    return outputDict


def Eigen(hamiltonian, basisStates, diagElements=[]):
    if len(diagElements) != 0:
        matrix = base.OperatorMatrix(basisStates, hamiltonian) + np.diag(diagElements)
    else:
        matrix = base.OperatorMatrix(basisStates, hamiltonian)
    eigenValues, eigenStates = scipy.linalg.eigh(matrix)
    eigenVecs = [{} for _ in eigenValues]
    for (j, vector) in enumerate(eigenStates):
        for (i, c_i) in enumerate(vector):
            if c_i == 0:
                continue
            multipliedDict = {k: c_i * v for (k, v) in basisStates[i].items()}
            eigenVecs[j] = AddDicts(eigenVecs[j], multipliedDict)
    return eigenValues, eigenVecs
