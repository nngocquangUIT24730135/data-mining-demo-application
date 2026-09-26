from __future__ import annotations

from itertools import product
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm, StepLogger
from datamining_app.algorithms.dataset_utils import excluded_headers
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult
from datamining_app.fmt import fmt_prob


class RoughSetAlgorithm(BaseAlgorithm):
    name = "Rough Set"
    description = "Tính reduct, core và luật quyết định từ ma trận phân biệt."
    supports_predict = True
    param_schema = {
        "decision_attr": ParamDef(
            type="choice",
            options="headers",
            default=None,
            label_vi="Thuộc tính quyết định",
            label_en="Decision attribute",
        ),
        "condition_attrs": ParamDef(
            type="multiselect",
            options="headers",
            default=[],
            label_vi="Thuộc tính điều kiện (để trống = tất cả trừ quyết định)",
            label_en="Condition attributes (empty = all except decision)",
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._rules: list[dict[str, Any]] = []
        self._reducts: list[set[str]] = []
        self._condition: list[str] = []
        self._decision: str = ""

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        headers = excluded_headers(dataset, params)
        selected = [c for c in (params.get("condition_attrs") or []) if c in headers]
        decision = params.get("decision_attr") or None
        if decision not in headers:
            decision = None
        if selected and not decision:
            condition = list(selected)
        elif decision:
            condition = [c for c in selected if c != decision] or [h for h in headers if h != decision]
        else:
            decision = headers[-1]
            condition = selected or [h for h in headers if h != decision]
        if not condition:
            raise ValueError("Cần ít nhất một thuộc tính điều kiện.")

        objects = [
            {
                "id": _object_id(row, i),
                "attrs": {h: str(row.get(h, "")).strip() for h in condition},
                "decision": str(row.get(decision, "")).strip() if decision else _object_id(row, i),
            }
            for i, row in enumerate(dataset.rows, start=1)
        ]
        universe = [obj["id"] for obj in objects]
        info_system = decision is None
        log = self._new_logger()
        steps = _RoughSetSteps(log)
        steps.system_step(universe, condition, decision, info_system)

        ind_a = indiscernibility(objects, condition)
        steps.ind_step(ind_a)

        classes = sorted({obj["decision"] for obj in objects})
        approximations = {}
        pos_count = 0
        gamma = 0.0
        if not info_system:
            for cls in classes:
                target = {obj["id"] for obj in objects if obj["decision"] == cls}
                lower, upper = approximate(ind_a, target)
                boundary = upper - lower
                alpha = (len(lower) / len(upper)) if upper else 1.0
                approximations[cls] = {
                    "target": sorted(target),
                    "lower": sorted(lower),
                    "upper": sorted(upper),
                    "boundary": sorted(boundary),
                    "alpha": alpha,
                }
                pos_count += len(lower)
                steps.approximation_step(cls, approximations[cls])

            gamma = pos_count / len(objects) if objects else 0.0
            steps.dependency_step(gamma, pos_count, len(objects), decision)

        _, cells = discernibility_matrix(objects, condition, relative=not info_system)
        steps.matrix_step(cells, universe)

        clauses = unique_clauses(cells)
        absorbed = absorb_clauses(clauses)
        steps.function_step(absorbed)

        reducts = cnf_to_reducts(absorbed)
        core = set.intersection(*reducts) if reducts else set()
        steps.reduct_step(reducts, core)

        chosen = reducts[0] if reducts else set(condition)
        rules = [] if info_system else extract_rules(objects, chosen, decision)
        if not info_system:
            steps.rules_step(rules, chosen, decision)

        self._rules = rules
        self._reducts = reducts
        self._condition = condition
        self._decision = decision
        summary = (
            f"Tìm {len(reducts)} reduct, Core = {{{', '.join(sorted(core)) or '∅'}}}. "
            + (f"γ = {fmt_prob(gamma)}. {len(rules)} luật từ reduct {{{', '.join(sorted(chosen))}}}." if not info_system else "Bảng quan hệ (không có thuộc tính quyết định).")
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"decision_attr": decision, "condition_attrs": condition},
            steps=log.steps,
            output={
                "reducts": [sorted(r) for r in reducts],
                "core": sorted(core),
                "gamma": gamma,
                "approximations": approximations,
                "rules": rules,
                "matrix": cells,
                "universe": universe,
                "condition": condition,
                "decision": decision,
            },
            summary=summary,
        )
        return self._finish(result, trained=not info_system)

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        if not self._trained or not self._rules:
            raise ValueError("Chưa huấn luyện mô hình Tập thô.")
        normalized = {k: str(v).strip() for k, v in sample.items()}
        lines = ["Duyệt luật quyết định theo reduct:"]
        for rule in self._rules:
            match = all(normalized.get(k, "") == v for k, v in rule["conditions"].items())
            cond = " AND ".join(f"{k}={v}" for k, v in rule["conditions"].items())
            mark = "✓ khớp" if match else "✗ không khớp"
            lines.append(f"  IF {cond} THEN {rule['decision']}  [{mark}]")
            if match:
                explanation = "\n".join(lines)
                return PredictResult(
                    label=rule["decision"],
                    explanation=explanation,
                    details={"rule": rule, "reducts": [sorted(r) for r in self._reducts]},
                    sample=normalized,
                )
        explanation = "\n".join(lines) + "\nKhông có luật nào khớp mẫu."
        return PredictResult(
            label="Không xác định",
            explanation=explanation,
            details={"matched": False},
            sample=normalized,
        )


class _RoughSetSteps:
    def __init__(self, log: StepLogger) -> None:
        self._log = log
        self._approx_explained = False

    def system_step(
        self,
        universe: list[str],
        condition: list[str],
        decision: str | None,
        info_system: bool,
    ) -> None:
        if info_system:
            description = (
                "① Ý tưởng cốt lõi của Rough Set (Tập thô):\n"
                "  Khi dữ liệu có mâu thuẫn / không đầy đủ thông tin,\n"
                "  không thể phân lớp chính xác 100%.\n"
                '  → Tập thô dùng xấp xỉ: "chắc chắn thuộc X" vs\n'
                '    "có thể thuộc X" thay vì phân lớp cứng.\n'
                "\n"
                "② Các thành phần:\n"
                f"  U = vũ trụ = {{{', '.join(universe)}}}\n"
                f"  A = thuộc tính = {{{', '.join(condition)}}}\n"
                "  Không có thuộc tính quyết định (tìm reduct của bảng quan hệ)."
            )
        else:
            description = (
                "① Ý tưởng cốt lõi của Rough Set (Tập thô):\n"
                "  Khi dữ liệu có mâu thuẫn / không đầy đủ thông tin,\n"
                "  không thể phân lớp chính xác 100%.\n"
                '  → Tập thô dùng xấp xỉ: "chắc chắn thuộc X" vs\n'
                '    "có thể thuộc X" thay vì phân lớp cứng.\n'
                "\n"
                "② Các thành phần:\n"
                f"  U = vũ trụ (tập tất cả đối tượng) = {{{', '.join(universe)}}}\n"
                f"  C = thuộc tính điều kiện = {{{', '.join(condition)}}}\n"
                f"  D = thuộc tính quyết định = {decision}"
            )
        self._log.add(
            "Hệ thông tin" if info_system else "Hệ quyết định — Đầu vào",
            description,
            {"universe": universe, "condition": condition, "decision": decision},
            "STEP_HEADER",
        )

    def ind_step(self, ind_a: list[set[str]]) -> None:
        description = (
            "[Mã giả Bước 1] IND(A) ← phân hoạch U theo A\n"
            "\n"
            "① Quan hệ không phân biệt IND(A):\n"
            "  u IND(A) v ⟺ a(u) = a(v) ∀a ∈ A\n"
            '  → u và v "trông giống nhau" theo mọi thuộc tính trong A.\n'
            "\n"
            "② Lớp tương đương [u]_A:\n"
            "  Nhóm tất cả đối tượng không phân biệt được với u.\n"
            "  → U / IND(A) = phân hoạch U thành các lớp tương đương.\n"
            '  → Đây là "độ phân giải" tối đa mà tập thuộc tính A cung cấp.\n'
            f"\n  {self._format_partition(ind_a)}"
        )
        self._log.add_table(
            "Quan hệ không phân biệt IND(A)",
            ["#", "Lớp tương đương"],
            [[str(i), "{" + ", ".join(sorted(p)) + "}"] for i, p in enumerate(ind_a, start=1)],
            ["right", "left"],
            description=description,
            partition=[sorted(p) for p in ind_a],
        )

    def approximation_step(self, cls: str, approximations_cls: dict[str, Any]) -> None:
        lower = approximations_cls["lower"]
        upper = approximations_cls["upper"]
        boundary = approximations_cls["boundary"]
        alpha = approximations_cls["alpha"]
        target = approximations_cls["target"]
        if not self._approx_explained:
            self._approx_explained = True
            concept = (
                "[Mã giả Bước 2] POS_A(D) ← xấp xỉ dưới\n"
                "\n"
                "① Tại sao cần xấp xỉ?\n"
                "  Khi 1 lớp tương đương chứa đối tượng từ nhiều lớp D\n"
                "  → không thể phân loại chính xác → dùng xấp xỉ.\n"
                "\n"
                '② Xấp xỉ dưới A̲X — "chắc chắn thuộc X":\n'
                "  = ∪{ [u]_A : [u]_A ⊆ X }\n"
                "  Chỉ lấy lớp tương đương NẰM HOÀN TOÀN trong X.\n"
                "\n"
                '③ Xấp xỉ trên ĀX — "có thể thuộc X":\n'
                "  = ∪{ [u]_A : [u]_A ∩ X ≠ ∅ }\n"
                "  Lấy mọi lớp tương đương CÓ GIAO với X.\n"
                "\n"
                '④ Biên BN(X) = ĀX \\ A̲X — "vùng mờ / không chắc":\n'
                '  BN(X) = ∅ → X là "tập rõ" (crisp): phân loại hoàn hảo\n'
                '  BN(X) ≠ ∅ → X là "tập thô" (rough): còn mâu thuẫn\n'
                "\n"
                "⑤ Độ chính xác xấp xỉ: α(X) = |A̲X| / |ĀX|\n"
                "  α = 1.00 → hoàn hảo | α < 1 → có vùng mờ\n"
                "\n"
            )
        else:
            concept = ""
        description = (
            f"{concept}"
            f"X = lớp '{cls}' = {{{', '.join(target)}}}\n"
            f"α_A(X) = |A̲X| / |ĀX| = {len(lower)}/{len(upper) or 1} = {fmt_prob(alpha)}"
        )
        self._log.add_table(
            f"Xấp xỉ tập X = lớp '{cls}'",
            ["Thành phần", "Tập hợp"],
            [
                ["Xấp xỉ dưới A̲X", "{" + ", ".join(lower or ["∅"]) + "}"],
                ["Xấp xỉ trên ĀX", "{" + ", ".join(upper or ["∅"]) + "}"],
                ["Biên BN(X)", "{" + ", ".join(boundary or ["∅"]) + "}"],
            ],
            description=description,
            level="SUCCESS" if alpha == 1 else "WARNING",
            conclusion=f"  → α_A(X) = {fmt_prob(alpha)}  ({'tập rõ' if alpha == 1 else 'tập thô'})",
            **approximations_cls,
        )

    def dependency_step(self, gamma: float, pos_count: int, n_objects: int, decision: str) -> None:
        self._log.add(
            "Mức độ phụ thuộc thuộc tính",
            (
                "① γ_A(D) — độ phụ thuộc của quyết định vào điều kiện:\n"
                "  γ = |POS_A(D)| / |U|  (tỉ lệ đối tượng chắc chắn phân lớp được).\n"
                f"  γ_A({decision}) = |POS_A| / |U| = {pos_count}/{n_objects} = {fmt_prob(gamma)}"
            ),
            {"gamma": gamma, "pos": pos_count},
        )

    def matrix_step(self, cells: list[dict[str, Any]], universe: list[str]) -> None:
        self._log.add(
            "Ma trận phân biệt M(u,v)",
            (
                "[Mã giả Bước 3] M(u,v) = {a ∈ C : a(u) ≠ a(v)}\n"
                "\n"
                '① M(u,v) — "chứng cứ phân biệt" cặp (u,v):\n'
                "  Tập thuộc tính mà u và v có GIÁ TRỊ KHÁC NHAU.\n"
                "  → Chỉ xét cặp (u,v) KHÁC lớp quyết định\n"
                "    (cùng lớp → không cần phân biệt).\n"
                "\n"
                "② Ý nghĩa: M(u,v) = tập thuộc tính TỐI THIỂU để\n"
                '  "thấy được" sự khác biệt giữa u và v.\n'
                "  → Nền tảng để xây hàm Boolean và tìm Reduct."
            ),
            {"cells": cells, "universe": universe},
        )

    def function_step(self, absorbed: list[set[str]]) -> None:
        self._log.add_table(
            "Hàm phân biệt f(C)",
            ["#", "Mệnh đề"],
            [[str(i), self._clause_text(c)] for i, c in enumerate(absorbed, start=1)],
            ["right", "left"],
            description=(
                "[Mã giả Bước 4] f(C) ← ∧ (∨ M(u,v))\n"
                "\n"
                "① Hàm Boolean f(C) — dạng CNF:\n"
                "  Mỗi ô M(u,v) → 1 mệnh đề OR: (a₁ ∨ a₂ ∨ ...)\n"
                '  Ý nghĩa: "Để phân biệt u và v, cần ÍT NHẤT 1 thuộc tính này."\n'
                "  f(C) = AND của tất cả mệnh đề = CNF\n"
                "  (Conjunctive Normal Form = tích của các tổng)\n"
                "\n"
                "② Rút gọn — Absorption:\n"
                "  Nếu mệnh đề A ⊆ mệnh đề B → loại B (A đã bao quát B rồi).\n"
                "  Ví dụ: (a) ⊆ (a ∨ b) → giữ (a), bỏ (a ∨ b).\n"
                "\n"
                "f(C) = " + (" ∧ ".join(self._clause_text(c) for c in absorbed) or "(rỗng)")
            ),
            clauses=[sorted(c) for c in absorbed],
        )

    def reduct_step(self, reducts: list[set[str]], core: set[str]) -> None:
        self._log.add_table(
            "Reduct (DNF) & Core",
            ["#", "Reduct"],
            [[str(i), "{" + ", ".join(sorted(r)) + "}"] for i, r in enumerate(reducts, start=1)],
            ["right", "left"],
            description=(
                "[Mã giả Bước 5] DNF tối giản → Reduct\n"
                "\n"
                "① Chuyển CNF → DNF (Disjunctive Normal Form):\n"
                "  CNF = tích của tổng: (a∨b) ∧ (a∨c) ∧ ...\n"
                "  DNF = tổng của tích: ab ∨ ac ∨ ...\n"
                "  → Nhân phân phối rồi hấp thụ (absorption).\n"
                "  Mỗi hạng từ (monomial) của DNF = 1 Reduct.\n"
                "\n"
                "② Reduct — tập thuộc tính tối giản:\n"
                "  Tập con của C vẫn giữ NGUYÊN khả năng phân biệt.\n"
                "  Tối giản: bỏ bất kỳ thuộc tính nào → mất phân biệt.\n"
                '  Có thể có nhiều Reduct (các "phương án" rút gọn khác nhau).\n'
                "\n"
                "③ Core — thuộc tính bắt buộc:\n"
                "  Core = ∩ (tất cả Reduct)\n"
                "  → Thuộc tính trong Core xuất hiện trong MỌI Reduct\n"
                "  → Không thể loại bỏ trong bất kỳ phương án nào.\n"
                "  Core = ∅ → không có thuộc tính nào là bắt buộc."
            ),
            level="SUCCESS",
            reducts=[sorted(r) for r in reducts],
            core=sorted(core),
            conclusion="  → Core = " + ("{" + ", ".join(sorted(core)) + "}" if core else "∅"),
        )

    def rules_step(self, rules: list[dict[str, Any]], chosen: set[str], decision: str) -> None:
        rule_rows = [
            [
                str(i),
                "IF "
                + " AND ".join(f"{k} = {v}" for k, v in rule["conditions"].items())
                + f" THEN {decision} = {rule['decision']}",
                ", ".join(rule.get("objects") or []),
            ]
            for i, rule in enumerate(rules, start=1)
        ]
        self._log.add_table(
            f"Sinh luật IF-THEN từ Reduct {{{', '.join(sorted(chosen))}}}",
            ["#", "Luật", "Đối tượng"],
            rule_rows,
            ["right", "left", "left"],
            description=(
                "[Mã giả Bước 7] Sinh luật IF-THEN từ Reduct nhỏ nhất\n"
                "\n"
                "① Nhóm các đối tượng có cùng giá trị theo Reduct.\n"
                "  Mỗi nhóm → 1 luật IF-THEN.\n"
                "\n"
                "② Phân loại luật:\n"
                '  Luật "thuần" (pure=True):  mọi obj trong nhóm cùng lớp\n'
                "    → Luật chắc chắn, độ tin cậy = 100%\n"
                '  Luật "không thuần":        có obj khác lớp trong nhóm\n'
                "    → Dùng lớp đa số, độ tin cậy < 100% (dữ liệu mâu thuẫn)"
            ),
            level="SUCCESS",
            decision_rules=rules,
            chosen_reduct=sorted(chosen),
            conclusion=f"  → {len(rules)} luật từ reduct đã chọn.",
        )

    @staticmethod
    def _format_partition(partition: list[set[str]]) -> str:
        parts = ["{" + ", ".join(sorted(p)) + "}" for p in partition]
        return "U / IND(A) = {" + ", ".join(parts) + "}"

    @staticmethod
    def _clause_text(clause: set[str]) -> str:
        if len(clause) == 1:
            return next(iter(clause))
        return "(" + " ∨ ".join(sorted(clause)) + ")"


def _object_id(row: dict[str, Any], index: int) -> str:
    for key, value in row.items():
        if key.lower() in {"id", "u", "rid", "tid", "pid", "object", "tuple"} and str(value).strip():
            return str(value).strip()
    return f"x{index}"


def indiscernibility(objects: list[dict[str, Any]], attrs: list[str]) -> list[set[str]]:
    buckets: dict[tuple[str, ...], set[str]] = {}
    for obj in objects:
        key = tuple(obj["attrs"].get(a, "") for a in attrs)
        buckets.setdefault(key, set()).add(obj["id"])
    return list(buckets.values())


def approximate(partition: list[set[str]], target: set[str]) -> tuple[set[str], set[str]]:
    lower: set[str] = set()
    upper: set[str] = set()
    for block in partition:
        if block <= target:
            lower |= block
        if block & target:
            upper |= block
    return lower, upper


def discernibility_matrix(
    objects: list[dict[str, Any]], attrs: list[str], relative: bool = True
) -> tuple[dict[tuple[str, str], set[str]], list[dict[str, Any]]]:
    matrix: dict[tuple[str, str], set[str]] = {}
    cells: list[dict[str, Any]] = []
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if relative and a["decision"] == b["decision"]:
                continue
            diff = {attr for attr in attrs if a["attrs"].get(attr) != b["attrs"].get(attr)}
            matrix[(a["id"], b["id"])] = diff
            cells.append({"i": a["id"], "j": b["id"], "attrs": sorted(diff)})
    return matrix, cells


def unique_clauses(cells: list[dict[str, Any]]) -> list[set[str]]:
    clauses = []
    seen = set()
    for cell in cells:
        attrs = frozenset(cell["attrs"])
        if not attrs or attrs in seen:
            continue
        seen.add(attrs)
        clauses.append(set(attrs))
    return clauses


def absorb_clauses(clauses: list[set[str]]) -> list[set[str]]:
    absorbed: list[set[str]] = []
    for clause in sorted(clauses, key=len):
        if any(existing <= clause for existing in absorbed):
            continue
        absorbed = [existing for existing in absorbed if not clause <= existing]
        absorbed.append(clause)
    return absorbed


def cnf_to_reducts(clauses: list[set[str]]) -> list[set[str]]:
    if not clauses:
        return []
    terms: list[set[str]] = [set()]
    for clause in clauses:
        nxt = []
        for term, lit in product(terms, clause):
            nxt.append(set(term) | {lit})
        terms = absorb_clauses(nxt)
    terms.sort(key=lambda t: (len(t), tuple(sorted(t))))
    return terms


def extract_rules(objects: list[dict[str, Any]], reduct: set[str], decision: str) -> list[dict[str, Any]]:
    attrs = sorted(reduct)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for obj in objects:
        key = tuple(obj["attrs"].get(a, "") for a in attrs)
        grouped.setdefault(key, []).append(obj)
    rules = []
    for key, members in grouped.items():
        decisions = {m["decision"] for m in members}
        majority = max(decisions, key=lambda d: sum(1 for m in members if m["decision"] == d))
        rules.append(
            {
                "conditions": dict(zip(attrs, key)),
                "decision": majority,
                "objects": [m["id"] for m in members],
                "pure": len(decisions) == 1,
                "decision_attr": decision,
            }
        )
    return rules
