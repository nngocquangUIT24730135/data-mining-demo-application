from datamining_app.algorithms.kmeans import KMeansAlgorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class KMeansPage(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(master, KMeansAlgorithm(), "kmeans", on_busy=on_busy, **kwargs)
