from datamining_app.data.embedded_db.seed import seed_database
import pytest


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "datasets.db"
    seed_database(path)
    return path
