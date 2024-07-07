import base
import itertools
import numpy as np

def ExpandBasis(basis, sector, eigvals, numNew):
    expandedBasis = {}
    diagElements = {}
    for newComb in itertools.product([0, 1], repeat=2*numNew):
        extraOcc = sum(newComb)
        extraMagz = sum(newComb[::2]) - sum(newComb[1::2])
        newSector = (sector[0] + extraOcc, sector[1] + extraMagz)
        expandedBasis[newSector] = []
        diagElements[newSector] = eigvals
        for (stateDict, energy) in zip(basis, eigvals):
            newKeys = [tuple(list(key) + list(newComb)) for key in stateDict.keys()]
            newState = {nk: v for (nk,v) in zip(newKeys, stateDict.values())}
            expandedBasis[newSector].append(newState)
    return expandedBasis, diagElements

def ClassifyBasis(basisStates):
    classifiedBasis = {}
    for stateDict in basisStates:
        totOcc = sum(list(stateDict.keys())[0])
        totMagz = (sum(list(stateDict.keys())[0][::2]) 
                   - sum(list(stateDict.keys())[0][1::2]))
        if (totOcc, totMagz) not in classifiedBasis.keys():
            classifiedBasis[(totOcc, totMagz)] = []

        classifiedBasis[(totOcc, totMagz)].append(stateDict)
    return classifiedBasis


def IterDiag(hamFlow, basisStates, numSitesFlow, retainSize, occupSectors = lambda x, N: True):
    assert len(numSitesFlow) == len(hamFlow)
    classifiedBasis = ClassifyBasis(basisStates)
    diagElementsClassified = {k: np.zeros(len(v)) for (k,v) in classifiedBasis.items()}

    numDiffFlow = [n2 - n1 for (n1, n2) in zip(numSitesFlow[:-1], numSitesFlow[1:])]
    eigvalFlow = [{} for _ in hamFlow]
    eigvecFlow = [{} for _ in hamFlow]

    for (i, hamiltonian) in enumerate(hamFlow):
        newClassifiedBasis = {}
        newDiagElementsClassified = {}
        for (sector, basis) in classifiedBasis.items():
            matrix = base.OperatorMatrix(basis, hamiltonian) + np.diag(diagElementsClassified[sector])
            eigvl, eigvc = base.Eigen(matrix, basis)
            eigvalFlow[i][sector] = eigvl
            eigvecFlow[i][sector] = eigvc
            if i == len(hamFlow) - 1:
                continue

            expandedBasis, newDiagElements = ExpandBasis(eigvc, sector, eigvl, numDiffFlow[i])
            assert expandedBasis.keys() == newDiagElements.keys()

            for (nk, basis) in expandedBasis.items():
                if nk not in newClassifiedBasis:
                    newClassifiedBasis[nk] = []
                    newDiagElementsClassified[nk] = []
                newClassifiedBasis[nk] = np.concatenate((newClassifiedBasis[nk], basis))
                newDiagElementsClassified[nk] = np.concatenate((newDiagElementsClassified[nk], newDiagElements[nk]))

        classifiedBasis = {}
        diagElementsClassified = {}
        for (sector, basis) in newClassifiedBasis.items():
            classifiedBasis[sector] = basis[np.argsort(newDiagElementsClassified[sector])][:retainSize]
            diagElementsClassified[sector] = np.sort(newDiagElementsClassified[sector])[:retainSize]

    return eigvalFlow, eigvecFlow
