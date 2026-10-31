# Hướng dẫn nhanh – Vision System

Tài liệu này tóm tắt cách cài đặt, chạy ứng dụng và bổ sung thư viện ảnh mẫu hình dạng.

## 1. Yêu cầu và thư viện cần cài

- Python **3.11 trở lên**.
- Gói chạy chính: FastAPI, Uvicorn, OpenCV (bản headless), NumPy, Pydantic Settings và python-multipart. Các gói này được khai báo trong `pyproject.toml` và sẽ tự cài cùng dự án.
- Gói phát triển/kiểm tra: `httpx`, `pytest`, `ruff` (tùy chọn `dev`).
- PyTorch (tùy chọn `neural`): chỉ cần khi muốn huấn luyện hoặc dùng mô hình neural hình dạng. Phát hiện cơ bản vẫn chạy bằng OpenCV khi không cài PyTorch.

## 2. Cài đặt và chạy

Mở PowerShell tại thư mục dự án:

```powershell
cd D:\Visual_System
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn visual_system.api:app --host 127.0.0.1 --port 8000 --reload
```

Mở `http://127.0.0.1:8000` để dùng dashboard; kiểm tra dịch vụ tại `http://127.0.0.1:8000/health`. Để huấn luyện mô hình neural, cài thêm và chạy:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[neural]"
.\.venv\Scripts\python.exe -m visual_system.train_shapes --epochs 60
```

## 3. Các tệp chương trình chính

| Tệp | Vai trò |
| --- | --- |
| `src/visual_system/api/app.py` | Tạo ứng dụng FastAPI và khởi tạo dịch vụ |
| `src/visual_system/api/routers/` | Các API health, camera, video, thư viện hình dạng và dashboard |
| `src/visual_system/pipeline.py` | Điều phối phát hiện, nhận dạng, tracking và kiểm tra |
| `src/visual_system/services/detector.py` | Phát hiện vùng vật thể màu bằng OpenCV |
| `src/visual_system/services/color_recognizer.py` | Nhận dạng màu chủ đạo |
| `src/visual_system/services/shape_recognizer.py` | Nhận dạng hình dạng và sử dụng mẫu/model tùy chọn |
| `src/visual_system/services/shape_library.py` | Lưu, đọc và phân loại ảnh mẫu hình dạng |
| `src/visual_system/services/video_processor.py` | Phân tích video theo từng frame |
| `src/visual_system/services/tracker.py` | Theo dõi vật thể qua các frame |
| `src/visual_system/config.py` | Cấu hình ngưỡng và đường dẫn kho mẫu/model |
| `src/visual_system/train_shapes.py` | Huấn luyện model hình dạng từ dữ liệu đã gán nhãn |
| `src/visual_system/dashboard.html` | Giao diện dashboard |

## 4. Chọn và thêm thư viện đầu vào (ảnh mẫu hình dạng)

“Thư viện đầu vào” ở đây là tập ảnh mẫu do người vận hành chọn và gán nhãn, không phải gói Python. Chọn ảnh rõ vật thể, tốt nhất mỗi ảnh có một vật thể 3D chính; dùng ảnh từ nhiều góc, khoảng cách, ánh sáng và nền gần điều kiện camera thực tế. Hạn chế ảnh trùng/gần trùng. Nhãn được hỗ trợ: `sphere`, `cylinder`, `cone`, `box`, `hexagonal_prism`.

1. Mở dashboard tại `http://127.0.0.1:8000`.
2. Tìm mục **Bổ sung mẫu hình dạng**, chọn nhãn đúng với vật thể.
3. Chọn một hoặc nhiều ảnh rồi tải lên; xem lại danh sách mẫu để kiểm tra nhãn. Xóa mẫu gán nhầm và tải lại với nhãn đúng.
4. Mẫu được lưu trong `data/shape_library/<nhãn>/`; metadata/đặc trưng nằm ở `data/shape_library.sqlite3`. Thêm mẫu không tự chạy huấn luyện.
5. Nếu chỉ dùng nhận dạng heuristic và thư viện mẫu, có thể chạy ứng dụng ngay. Muốn tạo/cập nhật model neural, cài tùy chọn `.[neural]`, sau đó chạy `python -m visual_system.train_shapes --epochs 60`. Model mặc định lưu tại `data/models/shape_net.pt` và được ứng dụng tự nạp nếu qua kiểm tra chất lượng.

Có thể thay đổi đường dẫn kho mẫu/model bằng các thiết lập `SHAPE_LIBRARY_PATH` và `NEURAL_SHAPE_MODEL_PATH` trong môi trường hoặc `.env`. API tương ứng để đọc kho là `GET /api/shapes/library`, thêm ảnh là `POST /api/shapes/samples` (multipart: `files`, `label`).

## 5. Luồng thao tác tổng quát

```text
Khởi động API → Mở dashboard → Chọn ảnh/video hoặc camera
→ Phát hiện vật thể → Nhận dạng màu/hình dạng → Theo dõi và kiểm tra
→ Xem kết quả trên dashboard
```

Khi cần cải thiện nhận dạng hình dạng: chuẩn bị ảnh có nhãn → tải vào thư viện → kiểm tra mẫu → huấn luyện tùy chọn → chạy lại ứng dụng và đánh giá với ảnh/video thực tế riêng.
