"""Rough Set console steps. Indiscernibility and reducts stay in ``algorithms.rough_set``."""
from __future__ import annotations

from typing import Any

from datamining_app.algorithms._logger import StepLogger
from datamining_app.core.models import PredictResult
from datamining_app.fmt import fmt_prob

class _RoughSetSteps:
    """Ghi bước Rough Set vào StepLogger từ phân hoạch, ma trận phân biệt và luật đã tính."""
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
                "  = ∪{ [u]_A : [u]_A ∩ X ≠ {} }\n"
                "  Lấy mọi lớp tương đương CÓ GIAO với X.\n"
                "\n"
                '④ Biên BN(X) = ĀX \\ A̲X — "vùng mờ / không chắc":\n'
                '  BN(X) = {} → X là "tập rõ" (crisp): phân loại hoàn hảo\n'
                '  BN(X) ≠ {} → X là "tập thô" (rough): còn mâu thuẫn\n'
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
                ["Xấp xỉ dưới A̲X", "{" + ", ".join(lower) + "}" if lower else "{}"],
                ["Xấp xỉ trên ĀX", "{" + ", ".join(upper) + "}" if upper else "{}"],
                ["Biên BN(X)", "{" + ", ".join(boundary) + "}" if boundary else "{}"],
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

    def function_step(self, original: list[set[str]], absorbed: list[set[str]]) -> None:
        original_text = " ∧ ".join(self._clause_text(c) for c in original) or "(rỗng)"
        absorbed_text = " ∧ ".join(self._clause_text(c) for c in absorbed) or "(rỗng)"
        self._log.add_table(
            "Hàm phân biệt f(C)",
            ["#", "Mệnh đề sau rút gọn"],
            [[str(i), self._clause_text(c)] for i, c in enumerate(absorbed, start=1)],
            ["right", "left"],
            description=(
                "[Mã giả Bước 4] f(C) ← ∧ (∨ M(u,v))\n"
                "\n"
                "① Hàm Boolean f(C) — dạng chuẩn tắc hội (CNF), trước khi rút gọn:\n"
                "  Mỗi ô M(u,v) khác rỗng → 1 tổng: (a₁ ∨ a₂ ∨ ...)\n"
                '  Ý nghĩa: "Để phân biệt u và v, cần ÍT NHẤT 1 thuộc tính này."\n'
                "  f(C) = hội của mọi tổng đó, bỏ ô trùng.\n"
                "  Dạng chuẩn tắc hội = tích của các tổng.\n"
                "\n"
                f"f(C) dạng chuẩn tắc hội (CNF) ban đầu = {original_text}\n"
                "\n"
                "② Rút gọn — luật hấp thụ:\n"
                "  Nếu mệnh đề A ⊆ mệnh đề B → loại B (A đã bao quát B rồi).\n"
                "  Ví dụ: (a) ⊆ (a ∨ b) → giữ (a), bỏ (a ∨ b).\n"
                "  Bảng dưới là các mệnh đề còn lại.\n"
                "\n"
                f"f(C) dạng chuẩn tắc hội (CNF) sau rút gọn = {absorbed_text}"
            ),
            clauses=[sorted(c) for c in absorbed],
            clauses_original=[sorted(c) for c in original],
        )

    def reduct_step(self, absorbed: list[set[str]], reducts: list[set[str]], core: set[str]) -> None:
        cnf_text = " ∧ ".join(self._clause_text(c) for c in absorbed) or "(rỗng)"
        dnf_text = self._dnf_text(reducts)
        self._log.add_table(
            "Dạng chuẩn tắc tuyển (DNF) & Core",
            ["#", "Reduct"],
            [[str(i), "{" + ", ".join(sorted(r)) + "}"] for i, r in enumerate(reducts, start=1)],
            ["right", "left"],
            description=(
                "[Mã giả Bước 5] Dạng chuẩn tắc hội đã rút gọn → dạng chuẩn tắc tuyển (DNF)\n"
                "\n"
                "① Lấy đúng dạng chuẩn tắc hội (CNF) sau rút gọn:\n"
                f"  f(C) = {cnf_text}\n"
                "\n"
                "② Nhân phân phối thành dạng chuẩn tắc tuyển (DNF):\n"
                "  Dạng chuẩn tắc tuyển = tổng của các tích.\n"
                "  Mỗi hạng tử = 1 reduct.\n"
                "\n"
                f"f(C) dạng chuẩn tắc tuyển (DNF) = {dnf_text}\n"
                "\n"
                "③ Reduct — tập thuộc tính tối giản:\n"
                "  Tập con của C vẫn giữ NGUYÊN khả năng phân biệt.\n"
                "  Tối giản: bỏ bất kỳ thuộc tính nào → mất phân biệt.\n"
                '  Có thể có nhiều reduct (các "phương án" rút gọn khác nhau).\n'
                "\n"
                "④ Core — thuộc tính bắt buộc:\n"
                "  Core = ∩ (tất cả reduct)\n"
                "  Thuộc tính trong Core xuất hiện trong mọi reduct.\n"
                "  Core = {} → không có thuộc tính nào là bắt buộc."
            ),
            level="SUCCESS",
            reducts=[sorted(r) for r in reducts],
            core=sorted(core),
            conclusion="  → Core = " + ("{" + ", ".join(sorted(core)) + "}" if core else "{}"),
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

    @staticmethod
    def _term_text(term: set[str]) -> str:
        attrs = sorted(term)
        if len(attrs) == 1:
            return attrs[0]
        return "(" + " ∧ ".join(attrs) + ")"

    @classmethod
    def _dnf_text(cls, terms: list[set[str]]) -> str:
        return " ∨ ".join(cls._term_text(term) for term in terms) or "(rỗng)"




def build_predict_explanation(
    sample: dict[str, Any],
    rules: list[dict[str, Any]],
    reducts: list[set[str]],
) -> PredictResult:
    """Match a sample against reduct rules and describe each comparison."""
    normalized = {k: str(v).strip() for k, v in sample.items()}
    lines = ["Duyệt luật quyết định theo reduct:"]
    for rule in rules:
        match = all(normalized.get(k, "") == v for k, v in rule["conditions"].items())
        cond = " AND ".join(f"{k}={v}" for k, v in rule["conditions"].items())
        mark = "✓ khớp" if match else "✗ không khớp"
        lines.append(f"  IF {cond} THEN {rule['decision']}  [{mark}]")
        if match:
            return PredictResult(
                label=rule["decision"],
                explanation="\n".join(lines),
                details={"rule": rule, "reducts": [sorted(r) for r in reducts]},
                sample=normalized,
            )
    return PredictResult(
        label="Không xác định",
        explanation="\n".join(lines) + "\nKhông có luật nào khớp mẫu.",
        details={"matched": False},
        sample=normalized,
    )


__all__ = ["_RoughSetSteps", "build_predict_explanation"]
