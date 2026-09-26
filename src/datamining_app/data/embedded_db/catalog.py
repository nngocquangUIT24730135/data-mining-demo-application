from __future__ import annotations

import json
from typing import Any

REGISTRY: list[dict[str, Any]] = [
    {
        "algorithm": "apriori",
        "table_name": "apriori_standard_9tx",
        "display_name": "Giao dịch chuẩn — 9 giao dịch, 5 items",
        "description": "9 giao dịch mua hàng với 5 mặt hàng (I1..I5) — bộ dữ liệu kinh điển để minh họa từng bước thuật toán Apriori.",
        "default_config": {"minsup": 0.2222, "minconf": 0.70},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_monkey_5tx",
        "display_name": "Giao dịch ký tự MONKEY — 5 giao dịch",
        "description": "5 giao dịch với các chữ cái M, O, N, K, E, Y — dùng để so sánh hiệu năng và tìm luật meta.",
        "default_config": {"minsup": 0.60, "minconf": 0.80},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_grocery_4tx",
        "display_name": "Cửa hàng bách hóa — 4 giao dịch",
        "description": "4 giao dịch mua sắm với 4 mặt hàng thực phẩm — bài tập nhập môn về thuật toán vét cạn và Apriori.",
        "default_config": {"minsup": 0.50, "minconf": 0.75},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_daily_basket_5tx",
        "display_name": "Giỏ hàng hàng ngày — 5 giao dịch",
        "description": "5 giao dịch mua sắm ngày thường với 5 mặt hàng (Bread, Cornflakes...) — minh họa Apriori và biểu diễn vector nhị phân.",
        "default_config": {"minsup": 0.60, "minconf": 0.75},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_numeric_4tx",
        "display_name": "Mã hóa số — 4 giao dịch",
        "description": "4 giao dịch với các mặt hàng dạng số (1..5) — ví dụ sinh tập ứng cử viên Cₖ và tập phổ biến Lₖ.",
        "default_config": {"minsup": 0.50, "minconf": 0.70},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_grocery_25tx",
        "display_name": "Bách hóa mở rộng — 25 giao dịch, 16 items",
        "description": "25 giao dịch mua sắm với 16 mặt hàng đa dạng — bài tập quy mô lớn hơn để kiểm tra hiệu năng Apriori.",
        "default_config": {"minsup": 0.25, "minconf": 0.70},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_butter_jam_5tx",
        "display_name": "Bơ - Mứt - Muffin — 5 giao dịch",
        "description": "5 giao dịch với 5 mặt hàng (Butter, Curd, Eggs, Jam, Muffin) — minh họa Apriori-TID, DHP và dựng cây FP-Tree.",
        "default_config": {"minsup": 0.60, "minconf": 0.75},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_letters_4tx",
        "display_name": "Chữ cái A-E — 4 giao dịch",
        "description": "4 giao dịch với 5 chữ cái — minh họa cơ sở mẫu điều kiện trong FP-Growth.",
        "default_config": {"minsup": 0.50, "minconf": 0.70},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_numeric_7tx",
        "display_name": "Mã số — 7 giao dịch",
        "description": "7 giao dịch với 4 item số (1..4) — bài tập FP-Growth trên dữ liệu số.",
        "default_config": {"minsup": 0.5714, "minconf": 0.70},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_beer_diaper_5tx",
        "display_name": "Bia và Tã lót — 5 giao dịch",
        "description": "5 giao dịch điển hình về phân tích giỏ hàng với 6 mặt hàng — bài tập toàn diện về Apriori.",
        "default_config": {"minsup": 0.40, "minconf": 0.70},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_theory_8tx",
        "display_name": "Lý thuyết 8 giao dịch — a,b,c,d,e",
        "description": "8 giao dịch lý thuyết với 5 item (a..e) — định nghĩa tần suất, độ hỗ trợ, độ tin cậy, lift và leverage.",
        "default_config": {"minsup": 0.375, "minconf": 0.80},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_fptree_5tx",
        "display_name": "FP-Tree mẫu — 5 giao dịch đa item",
        "description": "5 giao dịch với nhiều item để minh họa quá trình sắp xếp tần suất giảm dần và duyệt FP-Tree.",
        "default_config": {"minsup": 0.60, "minconf": 0.70},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_pharmacy_7tx",
        "display_name": "Thuốc bách hóa — 7 giao dịch",
        "description": "7 giao dịch mua thuốc với 5 loại (Aspirin, VitaminC...) — minh họa chuyển đổi sang ma trận nhị phân 0/1.",
        "default_config": {"minsup": 0.25, "minconf": 0.80},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_binary_matrix_8tx",
        "display_name": "Ma trận nhị phân — 8 giao dịch, 5 items",
        "description": "8 giao dịch biểu diễn dạng ma trận nhị phân với 5 item (i1..i5) — tính tập μ-frequent và hàm apriori_gen.",
        "default_config": {"minsup": 0.25, "minconf": 0.80},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_math_5tx",
        "display_name": "Bài tập toán học — 5 giao dịch, {a,b,c,d}",
        "description": "5 giao dịch với 4 item {a,b,c,d} — bài tập tìm tập phổ biến và luật kết hợp theo hướng toán học.",
        "default_config": {"minsup": 0.25, "minconf": 0.75},
    },
    {
        "algorithm": "apriori",
        "table_name": "apriori_math_10tx",
        "display_name": "Bài tập toán học — 10 giao dịch, {i1..i5}",
        "description": "10 giao dịch với 5 item {i1..i5} — tìm các tập 0.6-frequent và luật có độ tin cậy ≥ 0.75.",
        "default_config": {"minsup": 0.60, "minconf": 0.75},
    },
    {
        "algorithm": "rough_set",
        "table_name": "roughset_relation_7obj",
        "display_name": "Bảng quan hệ 5 thuộc tính — 7 bản ghi",
        "description": "7 đối tượng với 5 thuộc tính điều kiện {A,B,C,D,E} — xác định phân hoạch tương đương và tìm các tập rút gọn tối giản.",
        "default_config": {"condition_attrs": ["A", "B", "C", "D", "E"], "decision_attr": ""},
    },
    {
        "algorithm": "rough_set",
        "table_name": "roughset_relation_4obj",
        "display_name": "Bảng quan hệ 4 bản ghi — tìm Lõi",
        "description": "4 bản ghi với 4 thuộc tính {A,B,C,D} — ví dụ nhỏ gọn để xác định khả năng phân biệt và tìm Core.",
        "default_config": {"condition_attrs": ["A", "B", "C", "D"], "decision_attr": ""},
    },
    {
        "algorithm": "rough_set",
        "table_name": "roughset_playtennis_14obj",
        "display_name": "Hệ quyết định Chơi Tennis — 14 bản ghi",
        "description": "14 ngày quan sát thời tiết với 4 thuộc tính điều kiện và nhãn PlayTennis — minh họa vùng xấp xỉ và tập rút gọn quyết định.",
        "default_config": {
            "condition_attrs": ["Outlook", "Temperature", "Humidity", "Wind"],
            "decision_attr": "PlayTennis",
        },
    },
    {
        "algorithm": "rough_set",
        "table_name": "roughset_sunburn_8obj",
        "display_name": "Nguy cơ Rám nắng — 8 đối tượng",
        "description": "8 người với 4 đặc điểm (tóc, chiều cao, cân nặng, kem chống nắng) và nhãn Sunburn — xây dựng ma trận phân biệt và hàm Boolean.",
        "default_config": {
            "condition_attrs": ["Hair", "Height", "Weight", "Lotion"],
            "decision_attr": "Sunburn",
        },
    },
    {
        "algorithm": "rough_set",
        "table_name": "roughset_weather_decision_8obj",
        "display_name": "Bảng quyết định Thời tiết — 8 đối tượng",
        "description": "8 ngày với 3 thuộc tính (Outlook, Temperature, Humidity) và nhãn quyết định — xác định IND(C) và sinh luật quyết định tối giản.",
        "default_config": {
            "condition_attrs": ["Outlook", "Temperature", "Humidity"],
            "decision_attr": "Decision",
        },
    },
    {
        "algorithm": "id3",
        "table_name": "id3_buy_computer",
        "display_name": "Mua máy tính AllElectronics — 14 mẫu",
        "description": "14 khách hàng phân theo độ tuổi, thu nhập, tình trạng sinh viên và xếp loại tín dụng để dự đoán hành vi mua máy tính.",
        "default_config": {"max_depth": 5, "decision_attr": "buys_computer"},
    },
    {
        "algorithm": "id3",
        "table_name": "id3_weather_play",
        "display_name": "Dự báo thời tiết chơi thể thao — 14 ngày",
        "description": "14 quan sát thời tiết theo 4 yếu tố (trời, nhiệt độ, độ ẩm, gió) để phân loại có nên ra ngoài chơi thể thao không.",
        "default_config": {"max_depth": 5, "decision_attr": "Play"},
    },
    {
        "algorithm": "id3",
        "table_name": "id3_binary_simple",
        "display_name": "Phân lớp nhị phân đơn giản — 6 mẫu",
        "description": "6 mẫu nhị phân với 3 thuộc tính điều kiện để kiểm chứng từng bước tính Entropy và Information Gain cơ bản.",
        "default_config": {"max_depth": 3, "decision_attr": "Attr_D"},
    },
    {
        "algorithm": "naive_bayes",
        "table_name": "id3_buy_computer",
        "display_name": "Mua máy tính AllElectronics — NB cổ điển (14 mẫu)",
        "description": "Cùng bảng ID3 AllElectronics, Naïve Bayes cổ điển (không làm mịn) dự đoán buys_computer.",
        "default_config": {"decision_attr": "buys_computer"},
    },
    {
        "algorithm": "naive_bayes",
        "table_name": "id3_weather_play",
        "display_name": "Dự báo thời tiết — NB cổ điển (14 ngày)",
        "description": "Cùng bảng ID3 thời tiết, Naïve Bayes cổ điển (không làm mịn) dự đoán Play.",
        "default_config": {"decision_attr": "Play"},
    },
    {
        "algorithm": "naive_bayes",
        "table_name": "nb_buy_mobile",
        "display_name": "Dự đoán mua điện thoại — NB cổ điển (14 khách)",
        "description": "14 khách hàng — Naïve Bayes cổ điển (không làm mịn) dự đoán buy_mobile.",
        "default_config": {"decision_attr": "buy_mobile"},
    },
    {
        "algorithm": "naive_bayes_laplace",
        "table_name": "id3_buy_computer",
        "display_name": "Mua máy tính AllElectronics — NB Laplace (14 mẫu)",
        "description": "Cùng bảng ID3 AllElectronics, Naïve Bayes với làm mịn Laplace dự đoán buys_computer.",
        "default_config": {"laplace_alpha": 1.0, "decision_attr": "buys_computer"},
    },
    {
        "algorithm": "naive_bayes_laplace",
        "table_name": "id3_weather_play",
        "display_name": "Dự báo thời tiết — NB Laplace (14 ngày)",
        "description": "Cùng bảng ID3 thời tiết, Naïve Bayes với làm mịn Laplace dự đoán Play.",
        "default_config": {"laplace_alpha": 1.0, "decision_attr": "Play"},
    },
    {
        "algorithm": "naive_bayes_laplace",
        "table_name": "nb_buy_mobile",
        "display_name": "Dự đoán mua điện thoại — NB Laplace (14 khách)",
        "description": "14 khách hàng — Naïve Bayes Laplace dự đoán buy_mobile.",
        "default_config": {"laplace_alpha": 1.0, "decision_attr": "buy_mobile"},
    },
    {
        "algorithm": "cart_gini",
        "table_name": "id3_weather_play",
        "display_name": "Dự báo thời tiết — CART Gini (14 ngày)",
        "description": "Cùng bộ dữ liệu thời tiết ID3, xây cây theo Gini Index.",
        "default_config": {"max_depth": 5, "decision_attr": "Play"},
    },
    {
        "algorithm": "cart_gini",
        "table_name": "id3_buy_computer",
        "display_name": "Mua máy tính — CART Gini (14 mẫu)",
        "description": "Cùng bộ dữ liệu AllElectronics, xây cây theo Gini Index.",
        "default_config": {"max_depth": 5, "decision_attr": "buys_computer"},
    },
    {
        "algorithm": "cart_gini",
        "table_name": "id3_binary_simple",
        "display_name": "Phân lớp nhị phân — CART Gini (6 mẫu)",
        "description": "6 mẫu nhị phân đơn giản để minh họa từng bước tính Gini và ΔGini.",
        "default_config": {"max_depth": 3, "decision_attr": "Attr_D"},
    },
    {
        "algorithm": "kmeans",
        "table_name": "kmeans_2d_7pts",
        "display_name": "Tọa độ 2D — 7 điểm (k=2, Euclide)",
        "description": "7 điểm phân bố trong mặt phẳng 2D, gom thành 2 cụm bằng khoảng cách Euclide (khởi tạo trọng tâm ngẫu nhiên).",
        "default_config": {
            "k": 2,
            "distance": "euclidean",
            "init": "random",
            "max_iter": 20,
        },
    },
    {
        "algorithm": "kmeans",
        "table_name": "kmeans_student_scores",
        "display_name": "Điểm học sinh 4 bài kiểm tra (k=3, Manhattan)",
        "description": "10 học sinh với điểm Quiz1, Bài giữa kỳ, Quiz2 và Bài cuối kỳ — gom nhóm theo năng lực bằng khoảng cách Manhattan.",
        "default_config": {"k": 3, "distance": "manhattan", "init": "random", "max_iter": 20},
    },
    {
        "algorithm": "kmeans",
        "table_name": "kmeans_1d_website_age",
        "display_name": "Tuổi người dùng website — 19 giá trị 1D (k=2)",
        "description": "19 giá trị tuổi người dùng website — minh họa K-Means trên dữ liệu 1 chiều bằng khoảng cách Manhattan.",
        "default_config": {
            "k": 2,
            "distance": "manhattan",
            "init": "random",
            "max_iter": 30,
        },
    },
    {
        "algorithm": "kmeans",
        "table_name": "kmeans_2d_9pts",
        "display_name": "Tọa độ 2D — 9 điểm (k=3, Euclide)",
        "description": "9 điểm trong mặt phẳng 2D, phân thành 3 cụm — kiểm chứng từng vòng lặp gán điểm và cập nhật trọng tâm.",
        "default_config": {"k": 3, "distance": "euclidean", "init": "random", "max_iter": 20},
    },
    {
        "algorithm": "kmeans",
        "table_name": "kmeans_2d_4pts_simple",
        "display_name": "Tọa độ 2D — 4 điểm minh họa (k=2)",
        "description": "4 điểm cực đơn giản để minh họa ma trận phân hoạch U và quá trình cập nhật trọng tâm trong 1-2 vòng lặp.",
        "default_config": {"k": 2, "distance": "euclidean", "init": "random", "max_iter": 20},
    },
    {
        "algorithm": "kmeans",
        "table_name": "kmeans_2d_16pts",
        "display_name": "Tọa độ 2D — 16 điểm (k=3, Euclide)",
        "description": "16 điểm phân tán rõ thành 3 cụm tự nhiên — khởi tạo trọng tâm ngẫu nhiên, hội tụ trong số vòng giới hạn.",
        "default_config": {
            "k": 3,
            "distance": "euclidean",
            "init": "random",
            "max_iter": 20,
        },
    },
    {
        "algorithm": "kmeans",
        "table_name": "kmeans_2d_14pts",
        "display_name": "Tọa độ 2D — 14 điểm (k=3, Euclide)",
        "description": "14 điểm có cấu trúc phân cụm rõ ràng, minh họa K-Means phân hoạch và so sánh giữa 2 độ đo khoảng cách.",
        "default_config": {"k": 3, "distance": "euclidean", "init": "random", "max_iter": 20},
    },
]

for entry in list(REGISTRY):
    if entry["algorithm"] == "apriori":
        clone = dict(entry)
        clone["algorithm"] = "binary_vector"
        REGISTRY.append(clone)

DEFAULT_TABLES = {
    "apriori": "apriori_daily_basket_5tx",
    "binary_vector": "apriori_daily_basket_5tx",
    "rough_set": "roughset_relation_7obj",
    "id3": "id3_weather_play",
    "cart_gini": "id3_weather_play",
    "naive_bayes": "id3_weather_play",
    "naive_bayes_laplace": "id3_weather_play",
    "kmeans": "kmeans_2d_7pts",
}

ALGO_FALLBACK_PARAMS: dict[str, dict[str, Any]] = {
    "apriori": {"minsup": 0.50, "minconf": 0.75},
    "binary_vector": {"minsup": 0.50, "minconf": 0.75},
    "rough_set": {},
    "id3": {"max_depth": 5},
    "cart_gini": {"max_depth": 5},
    "naive_bayes": {},
    "naive_bayes_laplace": {"laplace_alpha": 1.0},
    "kmeans": {"k": 2, "distance": "euclidean", "max_iter": 20, "init": "random"},
}


def config_json(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False)


def params_for(algorithm: str, table_name: str) -> dict[str, Any]:
    for entry in REGISTRY:
        if entry["algorithm"] == algorithm and entry["table_name"] == table_name:
            return dict(entry["default_config"])
    return dict(ALGO_FALLBACK_PARAMS.get(algorithm, {}))
