from pathlib import Path

from datamining_app.algorithms.apriori import AprioriAlgorithm
from datamining_app.algorithms.binary_vector import BinaryVectorAlgorithm
from datamining_app.algorithms.cart_gini import CARTGiniAlgorithm
from datamining_app.algorithms.id3 import ID3Algorithm
from datamining_app.algorithms.kmeans import KMeansAlgorithm
from datamining_app.algorithms.rough_set import RoughSetAlgorithm
from datamining_app.data.csv_loader import CSVDataLoader
from datamining_app.data.embedded_db.catalog import ALL_REGISTRY, EXPORTED_TABLES, REGISTRY
from datamining_app.data.embedded_db.tables import TABLES
from datamining_app.data.preprocessor import Preprocessor, identifier_headers
from datamining_app.report_defaults import params_for

_DATA = Path(__file__).resolve().parents[1] / "data"
_ALGOS = {
    "apriori": AprioriAlgorithm,
    "binary_vector": BinaryVectorAlgorithm,
    "rough_set": RoughSetAlgorithm,
    "id3": ID3Algorithm,
    "cart_gini": CARTGiniAlgorithm,
    "kmeans": KMeansAlgorithm,
}


def test_half_the_datasets_live_in_csv():
    names = {path.stem for path in _DATA.glob("*.csv")}
    sqlite_names = {name for name, _, _ in TABLES}
    assert names == EXPORTED_TABLES
    assert not names & sqlite_names
    assert len(sqlite_names) == 16
    assert len(names) == 15
    assert all(entry["table_name"] not in EXPORTED_TABLES for entry in REGISTRY)


def test_exported_csv_runs_in_its_algorithm():
    loader = CSVDataLoader()
    prep = Preprocessor()
    ran = 0
    for entry in ALL_REGISTRY:
        name = entry["table_name"]
        algo_key = entry["algorithm"]
        if name not in EXPORTED_TABLES or algo_key not in _ALGOS:
            continue
        raw = loader.load(str(_DATA / f"{name}.csv"))
        dataset = prep.transform(raw, exclude_cols=identifier_headers(raw.headers), strip_whitespace=True)
        result = _ALGOS[algo_key]().run(dataset, params_for(algo_key, name))
        assert result.steps
        ran += 1
    assert ran >= len(EXPORTED_TABLES)
