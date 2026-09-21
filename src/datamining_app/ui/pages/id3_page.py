from datamining_app.algorithms.id3 import ID3Algorithm
from datamining_app.ui.pages.algorithm_page import AlgorithmPage


class ID3Page(AlgorithmPage):
    def __init__(self, master, on_busy=None, **kwargs) -> None:
        super().__init__(master, ID3Algorithm(), "id3", on_busy=on_busy, **kwargs)
