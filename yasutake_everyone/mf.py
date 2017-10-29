from sklearn.decomposition import NMF
import numpy as np
from numpy import nan
A = np.array([
        [5, nan, nan, nan,nan,nan],
        [4, nan, nan, 1,8,nan],
        [3, 1, nan, 5,9,7],
        [3, nan, nan, 4,7,8],
        [nan, 1, 6, 4,2,6],
        [nan,7,7,nan,3,0]
        ]
    )
model = NMF(n_components=50,solver='mu', tol=0.1,max_iter=100,init='random',random_state=0)
P = model.fit_transform(A)
Q = model.components_
class MatrixFactorization():
    def factorize(self,R):
        model = NMF(n_components=50,solver='mu', tol=0.1,max_iter=100,init='random',random_state=0)
        P = model.fit_transform(R)
        G = model.components_
        Q = G.transpose()
        return P,Q
