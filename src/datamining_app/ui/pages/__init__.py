from datamining_app.ui.pages.apriori_page import AprioriPage
from datamining_app.ui.pages.binary_vector_page import BinaryVectorPage
from datamining_app.ui.pages.home_page import HomePage
from datamining_app.ui.pages.id3_page import ID3Page
from datamining_app.ui.pages.kmeans_page import KMeansPage
from datamining_app.ui.pages.naive_bayes_page import NaiveBayesPage
from datamining_app.ui.pages.rough_set_page import RoughSetPage

PAGE_CLASSES = {
    "home": HomePage,
    "apriori": AprioriPage,
    "binary_vector": BinaryVectorPage,
    "rough_set": RoughSetPage,
    "id3": ID3Page,
    "naive_bayes": NaiveBayesPage,
    "kmeans": KMeansPage,
}

__all__ = ["PAGE_CLASSES"]
