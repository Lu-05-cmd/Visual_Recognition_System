# Vision System

Nền tảng hệ thống thị giác máy tính công nghiệp tự động phát hiện sản phẩm, nhận dạng màu sắc và hình dạng, theo dõi sản phẩm qua các frame và kiểm tra chất lượng.

## Thành phần hiện tại

- `ColorBlobDetector`: detector baseline bằng OpenCV, không cần model weights.
- `DominantColorRecognizer`: nhận dạng màu chính từ vùng sản phẩm.
- `ShapeRecognizer`: chỉ xuất nhãn khối 3D `sphere`, `cylinder`, `cone`, `box`; nhãn đường viền 2D không được xuất thay nhãn khối. Khi nhận dạng 3D chưa chắc chắn, đường viền (tròn/oval, tam giác, chữ nhật/oval hoặc đa giác) được dùng làm bằng chứng phụ cùng shading, vành tròn hoặc mặt khối; silhouette phẳng vẫn là `unknown`. Có thể dùng heuristic OpenCV và neural solid head nếu checkpoint đạt tiêu chí.
- `ShapeFeatureLibrary`: quản lý ảnh mẫu để huấn luyện; dữ liệu đường viền 2D cũ được giữ tương thích nhưng không tham gia nhận dạng runtime và không thể thêm mới qua API.
- `CentroidTracker`: gán `track_id` ổn định giữa các frame.
- `BasicInspector`: kiểm tra baseline bằng tỷ lệ cạnh ảnh.
- FastAPI dashboard và API demo.

Các service được tách riêng để thay detector bằng YOLO/ONNX hoặc nối camera công nghiệp mà không đổi API pipeline.

### Kiến trúc và ranh giới module

- `api/` chứa app factory và router riêng cho health, camera, video, kho hình dạng và dashboard. Có thể truyền settings/storage vào `create_app()` để kiểm thử; `visual_system.api:app` vẫn là entry point tương thích cho Uvicorn.
- `VisionPipeline` điều phối detector, nhận dạng màu/hình dạng, tracking và inspector. Hợp đồng `Detector` nằm ở `ports.py`; pipeline chỉ cần `detect(frame)` và không phụ thuộc thuật toán mask OpenCV.
- `services/` chứa các tích hợp và thuật toán độc lập; `models.py` chứa cấu trúc dữ liệu dùng chung. Có thể tiêm `detector_factory` vào `create_app()` để thay detector đồng nhất cho endpoint frame, camera và video; mặc định vẫn là detector OpenCV chạy offline.

Mọi lần tái cấu trúc cần giữ hành vi API hiện tại và có test cho app được tạo với storage riêng, để chạy nhiều cấu hình test không đụng dữ liệu mặc định trong `data/`.

## Cài đặt

Yêu cầu Python 3.11+.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Chạy API

Lưu ý: nên chạy bằng Python của virtual environment `.venv` để tránh lỗi `No module named uvicorn` khi máy đang có Anaconda/Miniconda trên PATH.

Chạy local trên máy hiện tại:

```powershell
cd D:\Visual_System
.\.venv\Scripts\python.exe -m uvicorn visual_system.api:app --host 127.0.0.1 --port 8000 --reload
```

Nếu muốn truy cập từ các thiết bị khác trên cùng mạng, chạy với host LAN:

```powershell
cd D:\Visual_System
.\.venv\Scripts\python.exe -m uvicorn visual_system.api:app --host 0.0.0.0 --port 8000 --reload
```

Mở `http://127.0.0.1:8000` để xem dashboard demo. Nếu đang dùng `0.0.0.0`, truy cập bằng địa chỉ IP máy chủ như `http://192.168.1.10:8000`. Health check ở `/health`, dữ liệu frame ở `/api/demo/frame`.

### Link chia sẻ công khai tạm thời

Ứng dụng hiện có thể truy cập tại [https://possess-improving-python-copper.trycloudflare.com](https://possess-improving-python-copper.trycloudflare.com). Đây là URL Cloudflare Quick Tunnel tạm thời; chỉ hoạt động khi server và tunnel còn chạy trên máy chủ. Khi tunnel dừng, URL có thể hết hiệu lực hoặc thay đổi.

## Phân tích video offline

Trong dashboard, chọn một file video ở mục **Phân tích video offline** rồi bấm **Phân tích video**. Hệ thống sẽ đọc toàn bộ video, chạy phát hiện/màu sắc/tracking/kiểm tra trên từng frame và tạo video MP4 mới có bounding box cùng nhãn kết quả.

Sau khi xử lý xong, dashboard phát video kết quả và hiển thị:

- số frame đã xử lý;
- số vật thể duy nhất được theo dõi;
- tổng lượt phát hiện qua các frame (một vật thể xuất hiện ở nhiều frame được tính nhiều lượt);
- số frame trùng liên tiếp được nhận ra bằng perceptual hash và bỏ qua bước nhận dạng để tránh đếm lại;
- số sản phẩm `PASS` và `FAIL`;
- thống kê màu sắc và hình dạng theo vật thể duy nhất.

API upload tương ứng là `POST /api/video/process` với multipart field `file`. Video kết quả được trả về qua `video_url`.

## Kho mẫu và huấn luyện neural hình dạng

Dashboard có mục **Bổ sung mẫu hình dạng** để gửi nhiều ảnh vật thể cùng nhãn. Ảnh gốc được lưu thành PNG trong `data/shape_library/<nhãn>/`, còn thông tin và đặc trưng được lưu trong `data/shape_library.sqlite3`. Có thể xóa riêng từng mẫu. Thêm ảnh vào kho không tự train lại model.

Kho mẫu có thể được xem qua `GET /api/shapes/library` và bổ sung qua `POST /api/shapes/samples` (multipart field `files` nhận nhiều ảnh, field `label` chỉ nhận `sphere`, `cylinder`, `cone`, `box`). Nhãn mẫu cũ `cube` và `cuboid` được gộp thành `box`. Các mẫu đường viền 2D đã lưu từ trước được giữ trong database để tương thích nhưng không được dùng trong nhận dạng runtime.

### Chuẩn bị dữ liệu

- Mỗi ảnh huấn luyện mới nên có **một vật thể 3D chính**, gán một nhãn: `sphere`, `cylinder`, `cone`, `box`.
- Thu thập góc nhìn, khoảng cách, vị trí trong khung, kích thước, màu sắc, ánh sáng và nền khác nhau. Tránh dùng nhiều frame liên tiếp gần như giống nhau làm mẫu độc lập.
- Ảnh render/xuất từ SolidWorks giúp bổ sung hình học và góc nhìn, nhưng nên trộn với ảnh camera/video thật. Bỏ thanh công cụ, con trỏ, chữ và viền cửa sổ nếu dùng ảnh chụp màn hình. Nền, bóng, màu và độ phản chiếu nên gần điều kiện camera thực tế.
- Chia dữ liệu theo lần chụp/điều kiện quay: giữ các video thật riêng để kiểm tra cuối, không đưa các frame của cùng một đoạn video vào cả train và validation. Nếu ảnh gần trùng nhau nằm ở hai tập, điểm validation có thể cao giả tạo.
- Mốc thu thập ban đầu để thử nghiệm là khoảng **100–200 ảnh đa dạng cho mỗi nhãn**. Đây không phải cam kết đạt 85%; độ chính xác phụ thuộc dữ liệu và điều kiện sử dụng. Ưu tiên bổ sung ảnh cho nhãn khối 3D đang thiếu mẫu.

### Train và checkpoint

Checkpoint hiện có hai đầu ra để giữ tương thích cấu trúc với model cũ; ứng dụng chỉ sử dụng đầu 3D (`sphere`, `cylinder`, `cone`, `box`), bỏ qua hoàn toàn đầu 2D và từ chối checkpoint không đạt quality gate cho nhánh khối. Mẫu 2D cũ được xem là ví dụ âm `none` cho đầu 3D khi huấn luyện; nhãn 2D không được xuất ra hoặc thống kê trong ứng dụng. Ảnh được cắt/đưa về kích thước chuẩn; khi train có tăng cường xoay, tịnh tiến, lật và thay đổi độ sáng/độ tương phản. Tập validation được tách tự động theo ảnh và có cân bằng trọng số lớp. Cài PyTorch một lần trong virtual environment rồi train:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[neural]"
.venv\Scripts\python.exe -m visual_system.train_shapes --epochs 60
```

Lệnh trên dùng kho `data/shape_library.sqlite3` và lưu model vào `data/models/shape_net.pt` theo mặc định.

Có thể đổi đường dẫn database, checkpoint, số epoch hoặc seed:

```powershell
.venv\Scripts\python.exe -m visual_system.train_shapes --help
.venv\Scripts\python.exe -m visual_system.train_shapes --epochs 80 --seed 42 --output data/models/shape_net.pt
```

Mỗi epoch in `train_loss`, `validation_loss`, độ chính xác và precision/recall theo từng lớp của các đầu tương thích 2D/3D. Cuối đợt train, checkpoint tốt nhất theo macro recall validation được lưu tại đường dẫn `--output` (mặc định `data/models/shape_net.pt`), kèm lịch sử, số mẫu và báo cáo chất lượng. Chỉ nhánh 3D đạt đủ mẫu train/validation và precision, recall, accuracy ít nhất 85% mới được ứng dụng nạp; nhánh 2D luôn bị bỏ qua. API tự nạp checkpoint hợp lệ; model không train lại mỗi lần chạy. Sau khi thêm ảnh mới, cần chạy lệnh train lại. Nếu train lỗi trước khi hoàn tất, checkpoint đích chưa được cập nhật.

Neural chỉ thay nhãn heuristic khi confidence đạt 0.85 và nhánh 3D vượt quality gate trong validation. Khi chưa cài PyTorch hoặc chưa có checkpoint đạt gate, API tiếp tục chạy heuristic khối 3D. Nếu checkpoint không đạt yêu cầu, báo cáo cuối lệnh train chỉ rõ lớp còn thiếu mẫu hoặc độ chính xác thấp; hãy bổ sung/kiểm tra nhãn và train lại thay vì chỉ tăng epoch.

Checkpoint trước phiên bản gộp nhãn không được nạp vì bộ nhãn khối đã thay đổi. Sau khi cập nhật mã, cần train lại để tạo checkpoint tương thích; nếu chưa đủ dữ liệu cho một nhánh, nhánh đó sẽ tiếp tục dùng heuristic.

### Đánh giá độ chính xác

Validation nội bộ giúp so sánh các lần train, nhưng cần tập test độc lập gồm ảnh/video thật từ các lần quay khác. Đánh giá nhãn khối 3D theo từng lớp và các trường hợp nhầm lẫn; không dùng độ chính xác nhánh 2D làm tiêu chí ứng dụng. Mục tiêu 85% chỉ được xác nhận khi đạt trên tập video thật holdout đại diện cho điều kiện vận hành. Các ảnh render SolidWorks và dữ liệu tăng cường không thay thế đánh giá thực tế.

Kết quả hình dạng trong video được làm ổn định bằng đồng thuận qua nhiều frame; track tồn tại tối đa 20 frame bị mất trước khi được đóng để tránh ghép nhầm vật thể khác.

## Kiểm thử

```powershell
pytest -q --basetemp=.pytest-tmp
ruff check .
```

`tests/test_shape_benchmark.py` kiểm tra các silhouette phẳng không bị xuất thành nhãn 3D, đánh giá nhận dạng khối 3D và định vị detector. `tests/test_shape_learning.py` giữ kiểm thử tương thích cho kho mẫu cũ; kho đó không tham gia nhận dạng runtime. `tests/test_video_processor.py` kiểm tra frame lặp, video đầu ra và track khi vật thể di chuyển. `tests/test_shape_library_integration.py` kiểm tra luồng fallback khi heuristic không nhận dạng được. Chạy lại toàn bộ suite sau mỗi thay đổi nhận dạng.

Kho mẫu dùng descriptor phiên bản 2 và được nâng cấp từ ảnh mẫu hiện có khi khởi động. Benchmark tổng hợp và ảnh train nội bộ không đại diện cho độ chính xác trên dây chuyền thật; cần đánh giá thêm bằng video thật đã gán nhãn, tách riêng khỏi ảnh huấn luyện.

## Cải tiến nhận dạng hình nón và hộp (v0.2)

### Vấn đề

- Hình nón ở các góc nhìn khác nhau bị nhãn là `unknown` thay vì `cone`
- Hình hộp tương tự được nhầm với `cylinder` hoặc `unknown`
- Nhãn dao động (oscillation) giữa các frame trước khi ổn định

### Giải pháp

1. **Tích hợp kho mẫu làm fallback** (`shape_recognizer.py`):
   - Gọi `ShapeFeatureLibrary.classify()` khi heuristic trả `unknown`
   - Guard `_is_flat_silhouette_only()` ngăn silhouette phẳng được nhãn là 3D
   - Truyền `shape_library_path` qua `VisionPipeline` → `api.create_app()` → `VideoProcessor`

2. **Cải tiến heuristic cho hình nón** (`_classify_solid_extracted()` → cone branch):
   - Chấp nhận `has_shading` ngoài `has_internal_facets` hoặc `_has_cone_shading()`
   - Giảm yêu cầu cho các hình tam giác có depth cue rõ

3. **Bộ classifier relaxed cho khoảng cách xa** (`ShapeFeatureLibrary.classify_relaxed()`):
   - Ngưỡng cao hơn (0.40 thay vì 0.32) cho vùng gray area
   - Chỉ dùng khi strict `classify()` trả `None`
   - Vẫn bảo vệ chống nhãn sai cho 2D

4. **Ổn định nhãn qua frame** (đã có `SHAPE_CONFIRMATION_FRAMES = 5`):
   - Nhãn chỉ lock sau 5 frame liên tiếp cùng nhãn
   - Trước đó hiển thị `unknown` để tránh dao động trong thống kê
   - Sau lock, nhãn được bảo vệ dù frame sau xảy ra lỗi tạm thời

### Kết quả thử nghiệm

- **Test suite**: 75 test pass (thêm 3 test integration)
- **Hình nón**: Cải thiện nhận dạng trên các góc nhìn khác nhau nhờ fallback library + heuristic đơn giản hơn
- **Hình hộp**: Kho mẫu cung cấp bằng chứng bổ sung khi heuristic không chắc
- **Silhouette 2D**: Vẫn từ chối được nhãn là 3D nhờ guard

### Dùng cải tiến

Không yêu cầu thay đổi code ứng dụng. API tự động sử dụng:
- Fallback library nếu kho mẫu được khởi tạo (mặc định từ `data/shape_library.sqlite3`)
- Heuristic cải tiến cho cone với shading lightweight
- Ổn định nhãn tự động qua 5 frame

## Hướng phát triển

1. Thay `ColorBlobDetector` bằng model detection đã huấn luyện.
2. Thêm adapter cho webcam, RTSP hoặc camera công nghiệp.
3. Bổ sung ROI, tiêu chí kiểm tra theo SKU và lưu kết quả vào database.
4. Thêm luồng video và giao diện biểu đồ năng suất/lỗi.
