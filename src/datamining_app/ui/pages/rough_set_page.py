from datamining_app.algorithms.rough_set import RoughSetAlgorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class RoughSetPage(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(master, RoughSetAlgorithm(), "rough_set", on_busy=on_busy, **kwargs)
