import iterDiag
import base
import time

numSteps = 5
numSitesFlow = [2]
basis = base.BasisStates(4)
hamFlow = [
        [("+-", [0, 2], 1.0), ("+-", [2, 0], 1.0), ("+-", [1, 3], 1.0), ("+-", [3, 1], 1.0), ("nn", [0, 1], 1.0), ("nn", [2, 3], 1.0)],
        ]
for step in range(1, numSteps+1):
    hamFlow.append(
            [("+-", [2 * step, 2 + 2 * step], 1.0), ("+-", [2 + 2 * step, 2 * step], 1.0), ("+-", [1 + 2 * step, 3 + 2 * step], 1.0), ("+-", [3 + 2 * step, 1 + 2 * step], 1.0), ("nn", [2 * step, 1 + 2 * step], 1.0), ("nn", [2 + 2 * step, 3 + 2 * step], 1.0)],
            )
    numSitesFlow.append(2 + step)
retainSize = 50
t = time.time()
e, v = iterDiag.IterDiag(hamFlow, basis, numSitesFlow, retainSize)
print(time.time() - t)
