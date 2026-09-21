from datamining_app.algorithms.apriori import AprioriAlgorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class AprioriPage(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(master, AprioriAlgorithm(), "apriori", on_busy=on_busy, **kwargs)
