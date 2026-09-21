from datamining_app.algorithms.apriori import AprioriAlgorithm
from datamining_app.algorithms.binary_vector import BinaryVectorAlgorithm
from datamining_app.algorithms.id3 import ID3Algorithm
from datamining_app.algorithms.kmeans import KMeansAlgorithm
from datamining_app.algorithms.naive_bayes import NaiveBayesAlgorithm
from datamining_app.algorithms.rough_set import RoughSetAlgorithm
from datamining_app.console.formatter import ConsoleFormatter, display_width
from datamining_app.data.embedded_db.catalog import DEFAULT_TABLES
from datamining_app.ui.step_slides import console_slides
from tests.helpers import load_report_dataset, report_params

ALGOS = {
    "apriori": AprioriAlgorithm,
    "binary_vector": BinaryVectorAlgorithm,
    "rough_set": RoughSetAlgorithm,
    "id3": ID3Algorithm,
    "naive_bayes": NaiveBayesAlgorithm,
    "kmeans": KMeansAlgorithm,
}


def _run(seeded_db, key: str):
    table = DEFAULT_TABLES[key]
    ds = load_report_dataset(seeded_db, table)
    return ALGOS[key]().run(ds, report_params(key, table))


def test_demo_slides_match_console_steps(seeded_db):
    for key in ALGOS:
        result = _run(seeded_db, key)
        slides = console_slides(result)
        assert len(slides) == len(result.steps)
        assert [s.step_number for s in slides] == [s.step_number for s in result.steps]
        assert slides == result.steps


def test_apriori_lists_all_frequent_itemsets(seeded_db):
    result = _run(seeded_db, "apriori")
    fmt = ConsoleFormatter()
    text = fmt.render_result(result, "apriori_daily_basket_5tx", 5)
    freq_step = next(s for s in result.steps if s.title.startswith("Tổng hợp"))
    rendered = fmt.render_step(freq_step)
    expected = result.output["supports"]
    for itemset in expected:
        assert itemset in rendered
    assert "Tổng hợp tập phổ biến" in text
    assert "Các luật đạt" not in text
    table_lines = [line for line in rendered.splitlines() if line.startswith(("┌", "├", "└", "│"))]
    assert table_lines
    assert len({display_width(line) for line in table_lines}) == 1


def test_console_tables_have_equal_row_width(seeded_db):
    fmt = ConsoleFormatter()
    for key in ALGOS:
        result = _run(seeded_db, key)
        for step in result.steps:
            text = fmt.render_step(step)
            table_lines = [line for line in text.splitlines() if line.startswith(("┌", "├", "└", "│"))]
            if len(table_lines) < 2:
                continue
            widths = {display_width(line) for line in table_lines}
            assert len(widths) == 1, f"{key} / {step.title}: {widths}"
