from datamining_app.algorithms.rough_set import RoughSetAlgorithm
from tests.helpers import load_report_dataset, report_params


def _reducts(result):
    return {frozenset(r) for r in result.output["reducts"]}


def test_roughset_7obj(seeded_db):
    ds = load_report_dataset(seeded_db, "roughset_relation_7obj")
    result = RoughSetAlgorithm().run(ds, report_params("rough_set", "roughset_relation_7obj"))
    assert {frozenset({"A", "C"}), frozenset({"D", "E"})} <= _reducts(result)
    assert set(result.output["core"]) == set()


def test_roughset_4obj(seeded_db):
    ds = load_report_dataset(seeded_db, "roughset_relation_4obj")
    result = RoughSetAlgorithm().run(ds, report_params("rough_set", "roughset_relation_4obj"))
    assert _reducts(result) == {frozenset({"A", "B"}), frozenset({"A", "D"})}
    assert set(result.output["core"]) == {"A"}


def test_roughset_playtennis_14(seeded_db):
    ds = load_report_dataset(seeded_db, "roughset_playtennis_14obj")
    result = RoughSetAlgorithm().run(ds, report_params("rough_set", "roughset_playtennis_14obj"))
    reducts = _reducts(result)
    assert frozenset({"Outlook", "Temperature", "Wind"}) in reducts or any(
        {"Outlook", "Wind"} <= set(r) for r in reducts
    )


def test_roughset_sunburn_8(seeded_db):
    ds = load_report_dataset(seeded_db, "roughset_sunburn_8obj")
    result = RoughSetAlgorithm().run(ds, report_params("rough_set", "roughset_sunburn_8obj"))
    assert "Hair" in set(result.output["core"])


def test_roughset_weather_8(seeded_db):
    ds = load_report_dataset(seeded_db, "roughset_weather_decision_8obj")
    result = RoughSetAlgorithm().run(ds, report_params("rough_set", "roughset_weather_decision_8obj"))
    reducts = _reducts(result)
    assert any({"Outlook", "Humidity"} <= set(r) for r in reducts) or frozenset(
        {"Outlook", "Humidity"}
    ) in reducts
