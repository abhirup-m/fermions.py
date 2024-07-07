import numpy as np

def BasisStates(numLevels, allowedOcc=None, allowedSz=None):
    basisStates = []
    for i in range(2**numLevels):
        binary = tuple([int(d) for d in bin(i)[2:].zfill(numLevels)])
        basisStates.append({binary: 1.0})
    return basisStates


def TransformBit(qubit, operator):
    assert operator in ("n", "h", "+", "-")
    if operator == "n":
        return qubit, qubit
    elif operator == "h":
        return qubit, 1 - qubit
    elif (operator == "+" and qubit == 0) or (operator == "-" and qubit == 1):
        return 1 - qubit, 1
    else:
        return qubit, 0


def ApplyOperator(operator, incomingState):
    assert max([max(positions) for _, positions, _ in operator]) < len(
        list(incomingState.keys())[0]
    )

    outgoingState = {}
    # loop over all operator tuples within operatorList
    for opType, opMembers, opStrength in operator:
        for incomingBasisState, coefficient in incomingState.items():
            newCoefficient = coefficient
            outgoingBasisState = list(incomingBasisState)

            # for each basis state, obtain a modified state after applying the operator tuple
            for siteIndex, operator in zip(reversed(opMembers), reversed(opType)):
                newQubit, factor = TransformBit(outgoingBasisState[siteIndex], operator)
                if factor == 0:
                    newCoefficient = 0
                    break
                # calculate the fermionic exchange sign by counting the number of
                # occupied states the operator has to "hop" over
                exchangeSign = (
                    (-1) ** sum(outgoingBasisState[:siteIndex])
                    if operator in ["+", "-"]
                    else 1
                )
                outgoingBasisState[siteIndex] = newQubit
                newCoefficient *= exchangeSign * factor
            outgoingBasisState = tuple(outgoingBasisState)
            if newCoefficient != 0:
                if outgoingBasisState in outgoingState.keys():
                    outgoingState[outgoingBasisState] += opStrength * newCoefficient
                else:
                    outgoingState[outgoingBasisState] = opStrength * newCoefficient

    return outgoingState


def OperatorMatrix(basisStates, operator):
    operatorMatrix = np.zeros((len(basisStates), len(basisStates)))

    for incomingIndex, incomingState in enumerate(basisStates):
        newState = ApplyOperator(operator, incomingState)

        for outgoingIndex, outgoingState in enumerate(basisStates):
            overlap = sum(
                newState[k] * outgoingState[k]
                for k in newState
                if k in outgoingState.keys()
            )
            operatorMatrix[outgoingIndex, incomingIndex] = overlap
    return operatorMatrix
