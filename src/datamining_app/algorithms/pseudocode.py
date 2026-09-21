PSEUDOCODE: dict[str, str] = {
    "apriori": """ALGORITHM APRIORI(D, minsup, minconf)
  INPUT  : D — tập giao dịch
           minsup — độ hỗ trợ tối thiểu
           minconf — độ tin cậy tối thiểu
  OUTPUT : Tập phổ biến L và luật kết hợp R

  1. C₁ ← tất cả item 1-phần tử
  2. L₁ ← {c ∈ C₁ : support(c) ≥ minsup}
  3. k ← 2
  4. LẶP khi L_{k-1} ≠ ∅:
       Cₖ ← apriori_gen(L_{k-1})   // join rồi prune
       VỚI mỗi ứng viên c ∈ Cₖ:
            đếm support(c) trên D
       Lₖ ← {c ∈ Cₖ : support(c) ≥ minsup}
       k ← k + 1
  5. L ← ∪ₖ Lₖ
  6. Sinh luật X ⇒ Y từ mỗi tập |itemset| ≥ 2 với conf ≥ minconf
  7. Trả về L, R
""",
    "binary_vector": """ALGORITHM BINARY-VECTOR(D, minsup, minconf)
  INPUT  : D — tập giao dịch
           minsup, minconf
  OUTPUT : Tập phổ biến F và luật R

  1. Xây ma trận ngữ cảnh nhị phân (O, I, R)
     vector(i)[t] = 1 nếu item i xuất hiện trong giao dịch t
  2. F₁ ← {i : |vector(i)| / |O| ≥ minsup}
  3. k ← 2
  4. LẶP khi F_{k-1} ≠ ∅:
       VỚI mỗi cặp (X, Y) ∈ F_{k-1} × F_{k-1}, |X ∪ Y| = k:
            v ← vector(X) AND vector(Y)
            NẾU |v| / |O| ≥ minsup → thêm X ∪ Y vào Fₖ
       k ← k + 1
  5. Sinh luật từ ∪ Fₖ với conf ≥ minconf
  6. Trả về F, R
""",
    "rough_set": """ALGORITHM ROUGH-SET(U, C, D)
  INPUT  : U — vũ trụ đối tượng
           C — thuộc tính điều kiện
           D — thuộc tính quyết định
  OUTPUT : Reducts, Core, luật quyết định

  1. IND(B) ← phân hoạch U theo bộ thuộc tính B ⊆ C
  2. POS_B(D) ← vùng dương (xấp xỉ dưới) của D theo B
  3. Lập ma trận phân biệt M(u,v) = {a ∈ C : a(u) ≠ a(v)}
     chỉ với cặp khác lớp quyết định
  4. f(C) ← ∧ (∨ M(u,v))   // hàm Boolean
  5. Đưa f(C) về DNF tối giản → mỗi hạng từ là một Reduct
  6. Core ← giao tất cả Reduct
  7. Sinh luật IF-THEN từ reduct nhỏ nhất
  8. Trả về Reducts, Core, luật
""",
    "id3": """ALGORITHM ID3(S, Attributes, Target)
  INPUT  : S — tập mẫu huấn luyện
           Attributes — danh sách thuộc tính điều kiện
           Target — thuộc tính quyết định
  OUTPUT : Cây quyết định T

  1. Tạo nút Root cho cây T
  2. NẾU tất cả mẫu trong S cùng lớp C:
       → Trả về nút lá với nhãn C
  3. NẾU Attributes = ∅:
       → Trả về nút lá với nhãn đa số trong S
  4. VỚI mỗi A ∈ Attributes:
       Tính Entropy(S)  = -Σ pᵢ log₂(pᵢ)
       Tính Info_A(S)   = Σ |Sᵥ|/|S| × Entropy(Sᵥ)
       Tính Gain(A, S)  = Entropy(S) - Info_A(S)
  5. A* ← argmax Gain(A, S)
  6. Đặt A* làm thuộc tính phân nhánh tại Root
  7. VỚI mỗi giá trị v của A*:
       Sᵥ ← {x ∈ S : x[A*] = v}
       NẾU Sᵥ = ∅ → Thêm lá với nhãn đa số trong S
       NGƯỢC LẠI → Thêm nhánh: ID3(Sᵥ, Attributes\\{A*}, Target)
  8. Trả về T
""",
    "naive_bayes": """ALGORITHM NAIVE-BAYES(D, Target, α)
  INPUT  : D — tập huấn luyện
           Target — thuộc tính lớp
           α — hệ số Laplace (thường = 1)
  OUTPUT : Mô hình P(C), P(xᵢ | C)

  1. VỚI mỗi lớp C:
       P(C) ← |D_C| / |D|
  2. VỚI mỗi lớp C, mỗi thuộc tính A, mỗi giá trị v:
       P(A=v | C) ← (count(A=v, C) + α) / (|D_C| + α · |A|)
  3. Dự đoán mẫu X = (x₁,...,xₙ):
       C* ← argmax_C  P(C) · Πᵢ P(xᵢ | C)
  4. Trả về C* và xác suất hậu nghiệm đã chuẩn hóa
""",
    "kmeans": """ALGORITHM K-MEANS(D, k, distance, max_iter)
  INPUT  : D — tập điểm dữ liệu
           k — số cụm
           distance — "euclidean" | "manhattan"
           max_iter — số vòng lặp tối đa
  OUTPUT : Phân hoạch {C₁,...,Cₖ} và trọng tâm {μ₁,...,μₖ}

  1. Khởi tạo k trọng tâm μ₁,...,μₖ  (random | first_k | manual)
  2. LẶP (t = 1 → max_iter):
     a. [Gán] VỚI mỗi xᵢ ∈ D:
              cluster(xᵢ) ← argminⱼ distance(xᵢ, μⱼ)
     b. [Cập nhật] VỚI mỗi cụm Cⱼ:
              μⱼ ← mean(xᵢ : cluster(xᵢ) = j)
     c. NẾU phân hoạch không đổi → DỪNG (hội tụ)
  3. Trả về {C₁,...,Cₖ}, {μ₁,...,μₖ}

  Hàm khoảng cách:
    Euclide  : d(x,y) = √(Σ (xᵢ-yᵢ)²)
    Manhattan: d(x,y) = Σ |xᵢ-yᵢ|
""",
}
