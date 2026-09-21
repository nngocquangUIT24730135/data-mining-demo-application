from datamining_app.algorithms.apriori import AprioriAlgorithm
from datamining_app.algorithms.binary_vector import BinaryVectorAlgorithm
from datamining_app.algorithms.id3 import ID3Algorithm
from datamining_app.algorithms.kmeans import KMeansAlgorithm
from datamining_app.algorithms.naive_bayes import NaiveBayesAlgorithm
from datamining_app.algorithms.rough_set import RoughSetAlgorithm

ALGORITHMS = {
    "apriori": AprioriAlgorithm,
    "binary_vector": BinaryVectorAlgorithm,
    "rough_set": RoughSetAlgorithm,
    "id3": ID3Algorithm,
    "naive_bayes": NaiveBayesAlgorithm,
    "kmeans": KMeansAlgorithm,
}

__all__ = [
    "ALGORITHMS",
    "AprioriAlgorithm",
    "BinaryVectorAlgorithm",
    "ID3Algorithm",
    "KMeansAlgorithm",
    "NaiveBayesAlgorithm",
    "RoughSetAlgorithm",
]
