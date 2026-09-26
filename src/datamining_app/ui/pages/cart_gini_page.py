from datamining_app.algorithms.cart_gini import CARTGiniAlgorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class CARTGiniPage(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(master, CARTGiniAlgorithm(), "cart_gini", on_busy=on_busy, **kwargs)
