import numpy as np
import scipy as sp
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
from sklearn.cross_validation import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.preprocessing import normalize
from pyemd import emd

class WordMoversKNN(KNeighborsClassifier):
    """K nearest neighbors classifier using the Word Mover's Distance.

    Parameters
    ----------
    
    W_embed : array, shape: (vocab_size, embed_size)
        Precomputed word embeddings between vocabulary items.
        Row indices should correspond to the columns in the bag-of-words input.

    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for Word Mover's Distance computation.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
    
    verbose : int, optional
        Controls the verbosity; the higher, the more messages. Defaults to 0.
        
    References
    ----------
    
    Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger
    From Word Embeddings To Document Distances
    The International Conference on Machine Learning (ICML), 2015
    http://mkusner.github.io/publications/WMD.pdf
    
    """
    _pairwise = False

    def __init__(self, W_embed, n_neighbors=1, n_jobs=1, verbose=False):
        self.W_embed = W_embed
        self.verbose = verbose
        super(WordMoversKNN, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs, metric='precomputed', algorithm='brute')

    def _wmd(self, i, row, X_train):
        """Compute the WMD between training sample i and given test row.
        
        Assumes that `row` and train samples are sparse BOW vectors summing to 1.
        """
        union_idx = np.union1d(X_train[i].indices, row.indices) - 1
        W_minimal = self.W_embed[union_idx]
        W_dist = euclidean_distances(W_minimal)
        bow_i = X_train[i, union_idx].A.ravel()
        bow_j = row[:, union_idx].A.ravel()
        return emd(bow_i, bow_j, W_dist)
    
    def _wmd_row(self, row, X_train):
        """Wrapper to compute the WMD of a row with all training samples.
        
        Assumes that `row` and train samples are sparse BOW vectors summing to 1.
        Useful for parallelization.
        """
        n_samples_train = X_train.shape[0]
        return [self._wmd(i, row, X_train) for i in range(n_samples_train)]

    def _pairwise_wmd(self, X_test, X_train=None):
        """Computes the word mover's distance between all train and test points.
        
        Parallelized over rows of X_test.
        
        Assumes that train and test samples are sparse BOW vectors summing to 1.
        
        Parameters
        ----------
        X_test: scipy.sparse matrix, shape: (n_test_samples, vocab_size)
            Test samples.
        
        X_train: scipy.sparse matrix, shape: (n_train_samples, vocab_size)
            Training samples. If `None`, uses the samples the estimator was fit with.
        
        Returns
        -------
        dist : array, shape: (n_test_samples, n_train_samples)
            Distances between all test samples and all train samples.
        
        """
        n_samples_test = X_test.shape[0]
        if X_train is None: X_train = self._fit_X
        if self.n_jobs == 1: dist = [ self._wmd_row( test_sample , X_train ) for test_sample in X_test ]
        else: dist = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)( delayed(self._wmd_row) (test_sample, X_train) for test_sample in X_test)
        return np.array(dist)

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : scipy sparse matrix, shape: (n_samples, n_features)
            Training data. 

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        return super(WordMoversKNN, self).fit(sp.sparse.csr_matrix(X), y)

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : scipy.sparse matrix, shape (n_test_samples, vocab_size)
            Test samples.

        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(sp.sparse.csr_matrix(X))
        return super(WordMoversKNN, self).predict(dist)
