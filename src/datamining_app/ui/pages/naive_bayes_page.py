from datamining_app.algorithms.naive_bayes import ClassicNaiveBayesAlgorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class NaiveBayesPage(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(master, ClassicNaiveBayesAlgorithm(), "naive_bayes", on_busy=on_busy, **kwargs)
