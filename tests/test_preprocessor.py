from datamining_app.core.models import Dataset
from datamining_app.data.preprocessor import Preprocessor


def test_exclude_and_normalize():
    ds = Dataset(
        "demo",
        ["tid", "Item", "Note"],
        [{"tid": "1", "Item": " Bread ", "Note": "X"}],
        "csv",
    )
    out = Preprocessor().transform(ds, exclude_cols=["tid"], strip_whitespace=True, lowercase=True)
    assert out.headers == ["Item", "Note"]
    assert out.rows[0]["Item"] == "bread"
    assert "tid" not in out.rows[0]
