from datamining_app.algorithms.apriori import AprioriAlgorithm
from datamining_app.algorithms.binary_vector import BinaryVectorAlgorithm
from datamining_app.algorithms.cart_gini import CARTGiniAlgorithm
from datamining_app.algorithms.id3 import ID3Algorithm
from datamining_app.algorithms.kmeans import KMeansAlgorithm
from datamining_app.algorithms.naive_bayes import (
    ClassicNaiveBayesAlgorithm,
    LaplaceBayesAlgorithm,
)
from datamining_app.algorithms.rough_set import RoughSetAlgorithm

ALGORITHMS = {
    "apriori": AprioriAlgorithm,
    "binary_vector": BinaryVectorAlgorithm,
    "rough_set": RoughSetAlgorithm,
    "id3": ID3Algorithm,
    "cart_gini": CARTGiniAlgorithm,
    "naive_bayes": ClassicNaiveBayesAlgorithm,
    "naive_bayes_laplace": LaplaceBayesAlgorithm,
    "kmeans": KMeansAlgorithm,
}

__all__ = [
    "ALGORITHMS",
    "AprioriAlgorithm",
    "BinaryVectorAlgorithm",
    "CARTGiniAlgorithm",
    "ClassicNaiveBayesAlgorithm",
    "ID3Algorithm",
    "KMeansAlgorithm",
    "LaplaceBayesAlgorithm",
    "RoughSetAlgorithm",
]
