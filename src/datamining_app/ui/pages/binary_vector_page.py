from datamining_app.algorithms.binary_vector import BinaryVectorAlgorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class BinaryVectorPage(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(master, BinaryVectorAlgorithm(), "binary_vector", on_busy=on_busy, **kwargs)
