from __future__ import annotations

from typing import Any

# Each table: (table_name, headers, rows) where rows are tuples matching headers.


def _txn(rows: list[tuple[str, str]]) -> tuple[list[str], list[tuple[Any, ...]]]:
    return ["tid", "items"], rows


TABLES: list[tuple[str, list[str], list[tuple[Any, ...]]]] = [
    (
        "apriori_standard_9tx",
        *_txn(
            [
                ("T100", "I1, I2, I5"),
                ("T200", "I2, I4"),
                ("T300", "I2, I3"),
                ("T400", "I1, I2, I4"),
                ("T500", "I1, I3"),
                ("T600", "I2, I3"),
                ("T700", "I1, I3"),
                ("T800", "I1, I2, I3, I5"),
                ("T900", "I1, I2, I3"),
            ]
        ),
    ),
    (
        "apriori_monkey_5tx",
        *_txn(
            [
                ("T100", "M, O, N, K, E, Y"),
                ("T200", "D, O, N, K, E, Y"),
                ("T300", "M, A, K, E"),
                ("T400", "M, U, C, K, Y"),
                ("T500", "C, O, K, I, E"),
            ]
        ),
    ),
    (
        "apriori_grocery_4tx",
        *_txn(
            [
                ("100", "Bread, Cornflakes"),
                ("101", "Bread, Cornflakes, Jam"),
                ("102", "Bread, Milk"),
                ("103", "Cornflakes, Jam, Milk"),
            ]
        ),
    ),
    (
        "apriori_daily_basket_5tx",
        *_txn(
            [
                ("T1", "Bread, Cornflakes, Eggs, Jam"),
                ("T2", "Bread, Cornflakes, Jam"),
                ("T3", "Bread, Milk, Tea"),
                ("T4", "Bread, Jam, Milk"),
                ("T5", "Cornflakes, Jam, Milk"),
            ]
        ),
    ),
    (
        "apriori_numeric_4tx",
        *_txn(
            [
                ("T1", "1, 3, 4"),
                ("T2", "2, 3, 5"),
                ("T3", "1, 2, 3, 5"),
                ("T4", "2, 5"),
            ]
        ),
    ),
    (
        "apriori_grocery_25tx",
        *_txn(
            [
                ("1", "Biscuit, Bournvita, Butter, Cornflakes, Tape"),
                ("2", "Bournvita, Bread, Butter, Cornflakes"),
                ("3", "Butter, Coffee, Chocolate, Eggs, Jam"),
                ("4", "Bournvita, Butter, Cornflakes, Bread, Eggs"),
                ("5", "Bournvita, Bread, Coffee, Chocolate, Eggs"),
                ("6", "Jam, Sugar"),
                ("7", "Biscuit, Bournvita, Butter, Cornflakes, Jam"),
                ("8", "Curd, Jam, Sugar"),
                ("9", "Bournvita, Bread, Butter, Coffee, Cornflakes"),
                ("10", "Bournvita, Bread, Coffee, Chocolate, Eggs"),
                ("11", "Bournvita, Butter, Eggs"),
                ("12", "Bournvita, Butter, Cornflakes, Chocolate, Eggs"),
                ("13", "Biscuit, Bournvita, Bread"),
                ("14", "Bread, Butter, Coffee, Chocolate, Eggs"),
                ("15", "Coffee, Cornflakes"),
                ("16", "Chocolate"),
                ("17", "Chocolate, Curd, Eggs"),
                ("18", "Biscuit, Bournvita, Butter, Cornflakes"),
                ("19", "Bournvita, Bread, Coffee, Chocolate, Eggs"),
                ("20", "Butter, Coffee, Chocolate, Eggs"),
                ("21", "Jam, Sugar, Tape"),
                ("22", "Bournvita, Bread, Butter, Cornflakes"),
                ("23", "Coffee, Chocolate, Eggs, Jam, Juice"),
                ("24", "Juice, Milk, Rice"),
                ("25", "Rice, Soap, Sugar"),
            ]
        ),
    ),
    (
        "apriori_butter_jam_5tx",
        *_txn(
            [
                ("100", "Butter, Curd, Eggs, Jam"),
                ("200", "Butter, Curd, Jam"),
                ("300", "Butter, Muffin, Nuts"),
                ("400", "Butter, Jam, Muffin"),
                ("500", "Curd, Jam, Muffin"),
            ]
        ),
    ),
    (
        "apriori_letters_4tx",
        *_txn(
            [
                ("100", "A, C, D"),
                ("200", "B, C, E"),
                ("300", "A, B, C, E"),
                ("400", "B, E"),
            ]
        ),
    ),
    (
        "apriori_numeric_7tx",
        *_txn(
            [
                ("T1", "1, 2, 3, 4"),
                ("T2", "1, 2, 4"),
                ("T3", "1, 2"),
                ("T4", "2, 3, 4"),
                ("T5", "2, 3"),
                ("T6", "3, 4"),
                ("T7", "2, 4"),
            ]
        ),
    ),
    (
        "apriori_beer_diaper_5tx",
        *_txn(
            [
                ("1", "Bread, Milk"),
                ("2", "Bread, Diaper, Beer, Eggs"),
                ("3", "Milk, Diaper, Beer, Coke"),
                ("4", "Bread, Milk, Diaper, Beer"),
                ("5", "Bread, Milk, Diaper, Coke"),
            ]
        ),
    ),
    (
        "apriori_theory_8tx",
        *_txn(
            [
                ("1", "a, b, c"),
                ("2", "a, b, c, d, e"),
                ("3", "b"),
                ("4", "c, d, e"),
                ("5", "c"),
                ("6", "b, c, d"),
                ("7", "c, d, e"),
                ("8", "c, e"),
            ]
        ),
    ),
    (
        "apriori_fptree_5tx",
        *_txn(
            [
                ("1", "f, a, c, d, g, i, m, p"),
                ("2", "a, b, c, f, l, m, o"),
                ("3", "b, f, h, j, o"),
                ("4", "b, c, k, s, p"),
                ("5", "a, f, c, e, l, p, m, n"),
            ]
        ),
    ),
    (
        "apriori_pharmacy_7tx",
        *_txn(
            [
                ("T1", "Aspirin, VitaminC"),
                ("T2", "Aspirin, Sudafed"),
                ("T3", "Tylenol"),
                ("T4", "Aspirin, VitaminC, Sudafed"),
                ("T5", "Tylenol, Cepacol"),
                ("T6", "Aspirin, Cepacol"),
                ("T7", "Aspirin, VitaminC"),
            ]
        ),
    ),
    (
        "apriori_binary_matrix_8tx",
        ["tid", "i1", "i2", "i3", "i4", "i5"],
        [
            ("T1", 1, 1, 0, 0, 0),
            ("T2", 0, 1, 1, 0, 0),
            ("T3", 1, 0, 0, 0, 1),
            ("T4", 1, 0, 0, 0, 1),
            ("T5", 0, 1, 1, 0, 1),
            ("T6", 1, 1, 1, 1, 1),
            ("T7", 1, 1, 1, 0, 0),
            ("T8", 0, 1, 1, 1, 1),
        ],
    ),
    (
        "apriori_math_5tx",
        *_txn(
            [
                ("T1", "a, b, c"),
                ("T2", "a, b, d"),
                ("T3", "a, c, d"),
                ("T4", "b, c, d"),
                ("T5", "a, b"),
            ]
        ),
    ),
    (
        "apriori_math_10tx",
        *_txn(
            [
                ("T1", "i1"),
                ("T2", "i1, i2"),
                ("T3", "i1, i2, i3"),
                ("T4", "i2, i3"),
                ("T5", "i2, i3, i4"),
                ("T6", "i1, i2, i4"),
                ("T7", "i1, i2, i5"),
                ("T8", "i2, i3, i4"),
                ("T9", "i2, i3, i5"),
                ("T10", "i3, i4, i5"),
            ]
        ),
    ),
    (
        "roughset_relation_7obj",
        ["pid", "A", "B", "C", "D", "E"],
        [
            ("t1", "a1", "b1", "c1", "d1", "e1"),
            ("t2", "a2", "b2", "c2", "d2", "e1"),
            ("t3", "a1", "b2", "c2", "d1", "e2"),
            ("t4", "a2", "b2", "c1", "d2", "e2"),
            ("t5", "a1", "b1", "c1", "d1", "e1"),
            ("t6", "a1", "b1", "c1", "d1", "e1"),
            ("t7", "a1", "b2", "c2", "d1", "e2"),
        ],
    ),
    (
        "roughset_relation_4obj",
        ["pid", "A", "B", "C", "D"],
        [
            ("t1", "a1", "b1", "c1", "d1"),
            ("t2", "a1", "b2", "c1", "d2"),
            ("t3", "a2", "b1", "c1", "d1"),
            ("t4", "a2", "b2", "c1", "d2"),
        ],
    ),
    (
        "roughset_playtennis_14obj",
        ["RID", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"],
        [
            (1, "Sunny", "Hot", "High", "Weak", "No"),
            (2, "Sunny", "Hot", "High", "Strong", "No"),
            (3, "Overcast", "Hot", "High", "Weak", "Yes"),
            (4, "Rain", "Mild", "High", "Weak", "Yes"),
            (5, "Rain", "Cool", "Normal", "Weak", "Yes"),
            (6, "Rain", "Cool", "Normal", "Strong", "No"),
            (7, "Overcast", "Cool", "Normal", "Strong", "Yes"),
            (8, "Sunny", "Mild", "High", "Weak", "No"),
            (9, "Sunny", "Cool", "Normal", "Weak", "Yes"),
            (10, "Rain", "Mild", "Normal", "Weak", "Yes"),
            (11, "Sunny", "Mild", "Normal", "Strong", "Yes"),
            (12, "Overcast", "Mild", "High", "Strong", "Yes"),
            (13, "Rain", "Hot", "Normal", "Weak", "Yes"),
            (14, "Rain", "Mild", "High", "Strong", "No"),
        ],
    ),
    (
        "roughset_sunburn_8obj",
        ["pid", "Hair", "Height", "Weight", "Lotion", "Sunburn"],
        [
            ("x1", "Blond", "Average", "Light", "No", "Yes"),
            ("x2", "Blond", "Tall", "Average", "Yes", "No"),
            ("x3", "Dark", "Short", "Average", "Yes", "No"),
            ("x4", "Blond", "Short", "Average", "No", "Yes"),
            ("x5", "Dark", "Average", "Heavy", "No", "No"),
            ("x6", "Brown", "Tall", "Heavy", "No", "No"),
            ("x7", "Brown", "Average", "Heavy", "Yes", "No"),
            ("x8", "Blond", "Average", "Heavy", "No", "Yes"),
        ],
    ),
    (
        "roughset_weather_decision_8obj",
        ["pid", "Outlook", "Temperature", "Humidity", "Decision"],
        [
            ("u1", "Sunny", "Hot", "High", "No"),
            ("u2", "Sunny", "Hot", "Normal", "No"),
            ("u3", "Overcast", "Hot", "High", "Yes"),
            ("u4", "Rain", "Mild", "High", "Yes"),
            ("u5", "Rain", "Cool", "Normal", "Yes"),
            ("u6", "Rain", "Cool", "Normal", "No"),
            ("u7", "Overcast", "Cool", "Normal", "Yes"),
            ("u8", "Sunny", "Mild", "High", "No"),
        ],
    ),
    (
        "id3_buy_computer",
        ["RID", "age", "income", "student", "credit_rating", "buys_computer"],
        [
            (1, "youth", "high", "no", "fair", "no"),
            (2, "youth", "high", "no", "excellent", "no"),
            (3, "middle_aged", "high", "no", "fair", "yes"),
            (4, "senior", "medium", "no", "fair", "yes"),
            (5, "senior", "low", "yes", "fair", "yes"),
            (6, "senior", "low", "yes", "excellent", "no"),
            (7, "middle_aged", "low", "yes", "excellent", "yes"),
            (8, "youth", "medium", "no", "fair", "no"),
            (9, "youth", "low", "yes", "fair", "yes"),
            (10, "senior", "medium", "yes", "fair", "yes"),
            (11, "youth", "medium", "yes", "excellent", "yes"),
            (12, "middle_aged", "medium", "no", "excellent", "yes"),
            (13, "middle_aged", "high", "yes", "fair", "yes"),
            (14, "senior", "medium", "no", "excellent", "no"),
        ],
    ),
    (
        "id3_weather_play",
        ["Instance", "Outlook", "Temperature", "Humidity", "Windy", "Play"],
        [
            (1, "sunny", "hot", "high", "false", "No"),
            (2, "sunny", "hot", "high", "true", "No"),
            (3, "overcast", "hot", "high", "false", "Yes"),
            (4, "rainy", "mild", "high", "false", "Yes"),
            (5, "rainy", "cool", "normal", "false", "Yes"),
            (6, "rainy", "cool", "normal", "true", "No"),
            (7, "overcast", "cool", "normal", "true", "Yes"),
            (8, "sunny", "mild", "high", "false", "No"),
            (9, "sunny", "cool", "normal", "false", "Yes"),
            (10, "rainy", "mild", "normal", "false", "Yes"),
            (11, "sunny", "mild", "normal", "true", "Yes"),
            (12, "overcast", "mild", "high", "true", "Yes"),
            (13, "overcast", "hot", "normal", "false", "Yes"),
            (14, "rainy", "mild", "high", "true", "No"),
        ],
    ),
    (
        "id3_binary_simple",
        ["Instance", "Attr_A", "Attr_B", "Attr_C", "Attr_D"],
        [
            (1, "T", "T", "T", "T"),
            (2, "T", "T", "F", "T"),
            (3, "F", "F", "T", "F"),
            (4, "T", "F", "T", "T"),
            (5, "F", "F", "F", "F"),
            (6, "T", "F", "F", "F"),
        ],
    ),
    (
        "nb_buy_mobile",
        ["RID", "age", "income", "Region", "credit_rating", "buy_mobile"],
        [
            (1, "<20", "high", "USA", "Low", "no"),
            (2, "<20", "high", "USA", "High", "no"),
            (3, "21...50", "high", "USA", "Low", "yes"),
            (4, ">50", "medium", "USA", "Low", "yes"),
            (5, ">50", "low", "PK", "Low", "yes"),
            (6, ">50", "low", "PK", "High", "no"),
            (7, "21...50", "low", "PK", "High", "yes"),
            (8, "<20", "medium", "USA", "Low", "no"),
            (9, "<20", "low", "PK", "Low", "yes"),
            (10, ">50", "medium", "PK", "Low", "yes"),
            (11, "<20", "medium", "PK", "High", "yes"),
            (12, "21...50", "medium", "USA", "High", "yes"),
            (13, "21...50", "high", "PK", "Low", "yes"),
            (14, ">50", "medium", "USA", "High", "no"),
        ],
    ),
    (
        "kmeans_2d_7pts",
        ["pid", "x", "y"],
        [
            ("1", 1.0, 1.0),
            ("2", 1.5, 2.0),
            ("3", 3.0, 4.0),
            ("4", 5.0, 7.0),
            ("5", 3.5, 5.0),
            ("6", 4.5, 5.0),
            ("7", 3.5, 4.5),
        ],
    ),
    (
        "kmeans_student_scores",
        ["pid", "Quiz1", "MSE", "Quiz2", "ESE"],
        [
            ("S1", 8, 20, 6, 45),
            ("S2", 6, 18, 7, 42),
            ("S3", 5, 15, 6, 35),
            ("S4", 4, 13, 5, 25),
            ("S5", 9, 21, 8, 48),
            ("S6", 7, 20, 9, 44),
            ("S7", 9, 17, 8, 49),
            ("S8", 8, 19, 7, 39),
            ("S9", 3, 14, 4, 22),
            ("S10", 6, 15, 7, 32),
        ],
    ),
    (
        "kmeans_1d_website_age",
        ["pid", "age"],
        [(str(i + 1), age) for i, age in enumerate(
            [15, 15, 16, 19, 19, 20, 20, 21, 22, 28, 35, 40, 41, 42, 43, 44, 60, 61, 65]
        )],
    ),
    (
        "kmeans_2d_9pts",
        ["pid", "x", "y"],
        [
            ("x1", 0.7, 0.45),
            ("x2", 2.8, 1.00),
            ("x3", 2.6, 1.00),
            ("x4", 1.0, 0.80),
            ("x5", 2.5, 1.20),
            ("x6", 1.3, 1.40),
            ("x7", 0.4, 0.70),
            ("x8", 1.7, 1.80),
            ("x9", 2.0, 2.00),
        ],
    ),
    (
        "kmeans_2d_4pts_simple",
        ["pid", "x", "y"],
        [
            ("x1", 1.0, 3.0),
            ("x2", 1.5, 3.2),
            ("x3", 1.3, 2.8),
            ("x4", 3.0, 1.0),
        ],
    ),
    (
        "kmeans_2d_16pts",
        ["pid", "x", "y"],
        [
            ("1", 6.8, 12.6),
            ("2", 0.8, 9.8),
            ("3", 1.2, 11.6),
            ("4", 2.8, 9.6),
            ("5", 3.8, 9.9),
            ("6", 4.4, 6.5),
            ("7", 4.8, 1.1),
            ("8", 6.0, 19.9),
            ("9", 6.2, 18.5),
            ("10", 7.6, 17.4),
            ("11", 7.8, 12.2),
            ("12", 6.6, 7.7),
            ("13", 8.2, 4.5),
            ("14", 8.4, 6.9),
            ("15", 9.0, 3.4),
            ("16", 9.6, 11.1),
        ],
    ),
    (
        "kmeans_2d_14pts",
        ["pid", "x", "y"],
        [
            ("v1", 1, 4),
            ("v2", 2, 3),
            ("v3", 2, 5),
            ("v4", 3, 4),
            ("v5", 3, 5),
            ("v6", 4, 2),
            ("v7", 4, 5),
            ("v8", 5, 1),
            ("v9", 5, 2),
            ("v10", 5, 3),
            ("v11", 5, 5),
            ("v12", 5, 6),
            ("v13", 6, 5),
            ("v14", 6, 2),
        ],
    ),
]
