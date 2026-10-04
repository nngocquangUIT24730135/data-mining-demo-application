# Ứng dụng Khai phá Dữ liệu

Ứng dụng desktop Python (tkinter) minh họa các giải thuật khai phá dữ liệu theo từng bước.
Console là trung tâm: bảng box-drawing, màu tag. Đồ thị chỉ mở popup cho ID3 và K-Means.

## Thông tin đồ án

| | |
| --- | --- |
| **Môn học** | Khai thác dữ liệu và truyền thông xã hội |
| **Đề tài** | Xây dựng ứng dụng mô phỏng và trực quan hóa các thuật toán trong môn Khai thác dữ liệu và truyền thông xã hội |
| **Giảng viên hướng dẫn** | ThS. Mai Xuân Hùng |

**Sinh viên thực hiện**

| MSSV | Họ và tên |
| --- | --- |
| 24730111 | Phan Doãn Luân |
| 24730095 | Phạm Đức Hải |
| 24730135 | Nguyễn Ngọc Quang |
| 24730133 | Võ Trường Phúc |

## Yêu cầu

- Python **3.11** trở lên
- [uv](https://docs.astral.sh/uv/) (trình quản lý môi trường và phụ thuộc)

## Cài đặt Python

1. Tải bộ cài từ [python.org/downloads](https://www.python.org/downloads/) (chọn Python 3.11 hoặc mới hơn).
2. Khi cài trên Windows, chọn **Add python.exe to PATH**.
3. Kiểm tra:

```powershell
python --version
```

Kết quả phải là `Python 3.11.x` hoặc cao hơn.

## Cài đặt uv

Trên Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Hoặc dùng winget:

```powershell
winget install --id=astral-sh.uv -e
```

Đóng rồi mở lại terminal, sau đó kiểm tra:

```powershell
uv --version
```

Hướng dẫn đầy đủ: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

## Cài đặt và chạy ứng dụng

Tại thư mục gốc của dự án:

```powershell
uv sync --group dev
uv run python -m datamining_app
```

Lệnh tương đương: `uv run datamining-app`.

## Kiểm thử

```powershell
uv run python -m pytest tests/ -v
```

Logic thuật toán viết bằng pure Python. `matplotlib` / `networkx` chỉ dùng cho popup trực quan hóa.
Xuất kết quả dạng TXT.

## Cơ sở dữ liệu SQLite

Bộ dữ liệu mẫu nằm trong file SQLite. File được tạo tự động lần đầu chạy ứng dụng (hoặc khi thiếu bảng):

```text
src/datamining_app/data/embedded_db/datasets.db
```

Có thể mở và kiểm tra file bằng [DB Browser for SQLite](https://sqlitebrowser.org/): **File → Open Database**, chọn `datasets.db` ở đường dẫn trên (tính từ thư mục gốc của dự án). Phần mềm cho phép xem bảng, sửa bản ghi và chạy câu SQL.

## Kiến trúc

```text
algorithms/   tính toán: run, predict, công thức
steps/        dựng chữ console từ dữ liệu đã tính
console/      vẽ bảng và tô màu
fmt.py        định dạng số
```

`algorithms/` gọi `steps/`. Package `steps/` không import module thuật toán.
