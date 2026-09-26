from datamining_app.algorithms.naive_bayes import LaplaceBayesAlgorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class LaplaceBayesPage(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(
            master, LaplaceBayesAlgorithm(), "naive_bayes_laplace", on_busy=on_busy, **kwargs
        )
