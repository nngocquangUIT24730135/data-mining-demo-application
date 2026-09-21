# Ứng dụng Khai phá Dữ liệu

Ứng dụng desktop Python (tkinter) minh họa các giải thuật khai phá dữ liệu theo từng bước.
Console là trung tâm: bảng box-drawing, màu tag. Đồ thị chỉ mở popup cho ID3 và K-Means.

## Cài đặt và chạy

```bash
uv sync --group dev
uv run python -m datamining_app
```

## Kiểm thử

```bash
uv run python -m pytest tests/ -v
```

Logic thuật toán viết bằng pure Python. `matplotlib` / `networkx` chỉ dùng cho popup trực quan hóa.
Xuất kết quả dạng TXT.
