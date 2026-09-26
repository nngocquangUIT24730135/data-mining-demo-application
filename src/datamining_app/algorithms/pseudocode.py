PSEUDOCODE: dict[str, str] = {
    "apriori": """ALGORITHM APRIORI(D, minsup, minconf)
  INPUT  : D — tập giao dịch
           minsup — độ hỗ trợ tối thiểu
           minconf — độ tin cậy tối thiểu
  OUTPUT : Tập phổ biến L và luật kết hợp R

  Bước 1: C₁ ← tất cả item 1-phần tử
  Bước 2: L₁ ← {c ∈ C₁ : support(c) ≥ minsup}
  Bước 3: k ← 2
  Bước 4: LẶP khi L_{k-1} ≠ ∅:
       Cₖ ← apriori_gen(L_{k-1})   // Join rồi Prune
       VỚI mỗi ứng viên c ∈ Cₖ:
            đếm support(c) trên D
       Lₖ ← {c ∈ Cₖ : support(c) ≥ minsup}
       k ← k + 1
  Bước 5: L ← ∪ₖ Lₖ
  Bước 6: Sinh luật X ⇒ Y từ mỗi tập |itemset| ≥ 2 với conf ≥ minconf
  Bước 7: Trả về L, R

  Công thức:
    support(X) = |{t ∈ D : X ⊆ t}| / |D|
    conf(X ⇒ Y) = support(X∪Y) / support(X)
""",
    "binary_vector": """ALGORITHM BINARY-VECTOR(D, minsup, minconf)
  INPUT  : D — tập giao dịch
           minsup, minconf
  OUTPUT : Tập phổ biến F và luật R

  Bước 1: Xây ma trận ngữ cảnh nhị phân (O, I, R)
     vector(i)[t] = 1 nếu item i xuất hiện trong giao dịch t
  Bước 2: F₁ ← {i : |vector(i)| / |O| ≥ minsup}
  Bước 3: k ← 2
  Bước 4: LẶP khi F_{k-1} ≠ ∅:
       VỚI mỗi cặp (X, Y) ∈ F_{k-1} × F_{k-1}, |X ∪ Y| = k:
            v ← vector(X) AND vector(Y)
            NẾU |v| / |O| ≥ minsup → thêm X ∪ Y vào Fₖ
       k ← k + 1
  Bước 5: Sinh luật từ ∪ Fₖ với conf ≥ minconf
  Bước 6: Trả về F, R
""",
    "rough_set": """ALGORITHM ROUGH-SET(U, C, D)
  INPUT  : U — vũ trụ đối tượng
           C — thuộc tính điều kiện
           D — thuộc tính quyết định
  OUTPUT : Reducts, Core, luật quyết định

  Bước 0: Hệ quyết định (U, C, D)
  Bước 1: IND(B) ← phân hoạch U theo bộ thuộc tính B ⊆ C
  Bước 2: POS_B(D) ← vùng dương (xấp xỉ dưới) của D theo B
  Bước 3: Lập ma trận phân biệt M(u,v) = {a ∈ C : a(u) ≠ a(v)}
     chỉ với cặp khác lớp quyết định
  Bước 4: f(C) ← ∧ (∨ M(u,v))   // hàm Boolean CNF
  Bước 5: Đưa f(C) về DNF tối giản → mỗi hạng từ là một Reduct
  Bước 6: Core ← giao tất cả Reduct
  Bước 7: Sinh luật IF-THEN từ reduct nhỏ nhất
  Bước 8: Trả về Reducts, Core, luật
""",
    "id3": """ALGORITHM ID3(S, Attributes, Target)
  INPUT  : S — tập mẫu huấn luyện
           Attributes — danh sách thuộc tính điều kiện
           Target — thuộc tính quyết định
  OUTPUT : Cây quyết định T

  Bước 1: Tạo nút Root cho cây T; tính Entropy(S) = -Σ pᵢ log₂(pᵢ)
  Bước 2: NẾU tất cả mẫu trong S cùng lớp C:
       → Trả về nút lá với nhãn C  (tập thuần khiết)
  Bước 3: NẾU Attributes = ∅ (hoặc đạt max_depth):
       → Trả về nút lá với nhãn đa số trong S
  Bước 4: VỚI mỗi thuộc tính A ∈ Attributes:
       Info_A(S)  = Σ (|Sᵥ|/|S|) × Entropy(Sᵥ)
       Gain(A, S) = Entropy(S) - Info_A(S)
  Bước 5: Chọn thuộc tính phân nhánh tối ưu:
       A* ← argmax_{A ∈ Attributes} Gain(A, S)
  Bước 6: Đặt A* làm thuộc tính phân nhánh tại Root
  Bước 7: VỚI mỗi giá trị v của A*:
       Sᵥ ← {x ∈ S : x[A*] = v}
       NẾU Sᵥ = ∅ → Thêm lá với nhãn đa số trong S
       NGƯỢC LẠI → Thêm nhánh: ID3(Sᵥ, Attributes \\ {A*}, Target)
  Bước 8: Trả về cây quyết định T
""",
    "naive_bayes": """ALGORITHM NAIVE-BAYES-NO-SMOOTHING(D, Target)
  INPUT  : D — tập huấn luyện
           Target — thuộc tính lớp
  OUTPUT : Mô hình P(C), P(xᵢ | C)  (không làm trơn)

  Bước 1: VỚI mỗi lớp C:
       P(C) ← |D_C| / |D|          // tiên nghiệm
  Bước 2: VỚI mỗi lớp C, mỗi thuộc tính A, mỗi giá trị v:
       P(A=v | C) ← count(A=v, C) / |D_C|
       // Nếu count = 0 → P = 0 → lớp C bị loại khi dự đoán
  Bước 3: Dự đoán mẫu X = (x₁,...,xₙ):
       C* ← argmax_C  P(C) · Πᵢ P(xᵢ | C)
  Bước 4: Trả về C* và xác suất hậu nghiệm đã chuẩn hóa
""",
    "naive_bayes_laplace": """ALGORITHM NAIVE-BAYES-LAPLACE(D, Target, α)
  INPUT  : D — tập huấn luyện
           Target — thuộc tính lớp
           α — hệ số Laplace (thường = 1)
  OUTPUT : Mô hình P(C), P(xᵢ | C)

  Bước 1: VỚI mỗi lớp C:
       P(C) ← |D_C| / |D|          // tiên nghiệm
  Bước 2: VỚI mỗi lớp C, mỗi thuộc tính A, mỗi giá trị v:
       P(A=v | C) ← (count(A=v, C) + α) / (|D_C| + α · |A|)
  Bước 3: Dự đoán mẫu X = (x₁,...,xₙ):
       C* ← argmax_C  P(C) · Πᵢ P(xᵢ | C)   // hậu nghiệm
  Bước 4: Trả về C* và xác suất hậu nghiệm đã chuẩn hóa
""",
    "cart_gini": """ALGORITHM CART-GINI(S, Attributes, Target)
  INPUT  : S — tập mẫu huấn luyện
           Attributes — danh sách thuộc tính điều kiện
           Target — thuộc tính quyết định
  OUTPUT : Cây quyết định T

  Bước 1: Tạo nút Root cho cây T; tính Gini(S) = 1 - Σ pᵢ²
  Bước 2: NẾU tất cả mẫu trong S cùng lớp C:
       → Trả về nút lá với nhãn C  (tập thuần khiết, Gini = 0)
  Bước 3: NẾU Attributes = ∅ (hoặc đạt max_depth):
       → Trả về nút lá với nhãn đa số trong S
  Bước 4: VỚI mỗi thuộc tính A ∈ Attributes:
       Gini_A(S) = Σ (|Sᵥ|/|S|) × Gini(Sᵥ)
       ΔGini(A)  = Gini(S) - Gini_A(S)
  Bước 5: Chọn thuộc tính phân nhánh tối ưu:
       A* ← argmax_{A ∈ Attributes} ΔGini(A)
  Bước 6: Đặt A* làm thuộc tính phân nhánh tại Root
  Bước 7: VỚI mỗi giá trị v của A*:
       Sᵥ ← {x ∈ S : x[A*] = v}
       NẾU Sᵥ = ∅ → Thêm lá với nhãn đa số trong S
       NGƯỢC LẠI → Thêm nhánh: CART-GINI(Sᵥ, Attributes \\ {A*}, Target)
  Bước 8: Trả về cây quyết định T
""",
    "kmeans": """ALGORITHM K-MEANS(D, k, distance, max_iter)
  INPUT  : D — tập điểm dữ liệu
           k — số cụm
           distance — "euclidean" | "manhattan"
           max_iter — số vòng lặp tối đa
  OUTPUT : Phân hoạch {C₁,...,Cₖ} và trọng tâm {μ₁,...,μₖ}

  Bước 1: Khởi tạo k trọng tâm μ₁,...,μₖ ngẫu nhiên từ các điểm trong D
  Bước 2: LẶP (t = 1 → max_iter):
     Bước 2a [Gán cụm] VỚI mỗi xᵢ ∈ D:
              cluster(xᵢ) ← argminⱼ distance(xᵢ, μⱼ)
     Bước 2b [Cập nhật trọng tâm] VỚI mỗi cụm Cⱼ:
              μⱼ ← mean({ xᵢ : cluster(xᵢ) = j })
     Bước 2c [Hội tụ] NẾU tất cả trọng tâm không đổi → DỪNG
     Bước 2d [Giới hạn] NẾU t = max_iter → DỪNG
  Bước 3: Trả về {C₁,...,Cₖ}, {μ₁,...,μₖ}

  Hàm khoảng cách:
    Euclide  : d(x,y) = √(Σ (xᵢ-yᵢ)²)   — tiêu chí SSE = Σ d²
    Manhattan: d(x,y) = Σ |xᵢ-yᵢ|       — tiêu chí SAE = Σ d
""",
}
