# Vision System

Nền tảng hệ thống thị giác máy tính công nghiệp tự động phát hiện sản phẩm, nhận dạng màu sắc và hình dạng, theo dõi sản phẩm qua các frame và kiểm tra chất lượng.

## Thành phần hiện tại

- `ColorBlobDetector`: detector baseline bằng OpenCV, không cần model weights.
- `DominantColorRecognizer`: nhận dạng màu chính từ vùng sản phẩm.
- `ShapeRecognizer`: nhận dạng riêng đường viền 2D và khối 3D. Kho mẫu 2D không được ghi đè loại khối; cone/cube chỉ được gán khi có gradient hoặc đường phân mặt bên trong phù hợp. Đây vẫn là heuristic OpenCV, độ chính xác phụ thuộc góc nhìn và ánh sáng.
- `ShapeFeatureLibrary`: lưu các mẫu hình có nhãn trong SQLite và nhận dạng few-shot bằng đặc trưng contour mở rộng (độ tròn, tỷ lệ, độ phức tạp, extent, solidity và Hu moments); ảnh mẫu cũ được nâng cấp đặc trưng khi khởi động.
- `CentroidTracker`: gán `track_id` ổn định giữa các frame.
- `BasicInspector`: kiểm tra baseline bằng tỷ lệ cạnh ảnh.
- FastAPI dashboard và API demo.

Các service được tách riêng để thay detector bằng YOLO/ONNX hoặc nối camera công nghiệp mà không đổi API pipeline.

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

Kho mẫu cũng có thể được dùng qua `GET /api/shapes/library` và `POST /api/shapes/samples` (multipart field `files` nhận nhiều ảnh và field `label` nhận hình dạng đã chọn). Nhãn `circle`, `oval`, `triangle`, `rectangle`, `square`, `pentagon`, `hexagon`, `polygon` là mẫu đường viền 2D; nhãn `sphere`, `cylinder`, `cone`, `cube`, `cuboid` là mẫu khối 3D (`cuboid` là khối hộp chữ nhật). Mỗi ảnh nên có đúng một vật thể, đủ sáng và nhiều góc nhìn/nền khác nhau.

### Chuẩn bị dữ liệu

- Mỗi ảnh nên có **một vật thể chính**, gán đúng một nhãn. Nhãn đường viền phẳng: `circle`, `oval`, `triangle`, `rectangle`, `square`, `pentagon`, `hexagon`, `polygon`. Nhãn khối: `sphere`, `cylinder`, `cone`, `cube`, `cuboid`.
- Thu thập góc nhìn, khoảng cách, vị trí trong khung, kích thước, màu sắc, ánh sáng và nền khác nhau. Tránh dùng nhiều frame liên tiếp gần như giống nhau làm mẫu độc lập.
- Ảnh render/xuất từ SolidWorks giúp bổ sung hình học và góc nhìn, nhưng nên trộn với ảnh camera/video thật. Bỏ thanh công cụ, con trỏ, chữ và viền cửa sổ nếu dùng ảnh chụp màn hình. Nền, bóng, màu và độ phản chiếu nên gần điều kiện camera thực tế.
- Chia dữ liệu theo lần chụp/điều kiện quay: giữ các video thật riêng để kiểm tra cuối, không đưa các frame của cùng một đoạn video vào cả train và validation. Nếu ảnh gần trùng nhau nằm ở hai tập, điểm validation có thể cao giả tạo.
- Mốc thu thập ban đầu để thử nghiệm là khoảng **100–200 ảnh đa dạng cho mỗi nhãn**. Đây không phải cam kết đạt 85%; độ chính xác phụ thuộc dữ liệu và điều kiện sử dụng. Tăng dữ liệu cho lớp đang thiếu trước (hiện `oval` chỉ có 2 ảnh).

### Train và checkpoint

Mạng CNN có hai đầu ra: đầu 2D học đường viền, đầu 3D học `sphere`, `cylinder`, `cone`, `cube`, `cuboid` và nhãn `none` cho ảnh chỉ có nhãn 2D. Ảnh được cắt/đưa về kích thước chuẩn; khi train có tăng cường xoay, tịnh tiến, lật và thay đổi độ sáng/độ tương phản. Tập validation được tách tự động theo ảnh và có cân bằng trọng số lớp. Cài PyTorch một lần trong virtual environment rồi train:

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

Mỗi epoch in `train_loss`, `validation_loss`, độ chính xác và precision/recall theo từng lớp riêng cho 2D và 3D. Cuối đợt train, checkpoint tốt nhất theo macro recall validation được lưu tại đường dẫn `--output` (mặc định `data/models/shape_net.pt`), kèm lịch sử, số mẫu và báo cáo lớp nào đủ điều kiện dùng. Checkpoint chỉ được kích hoạt cho một nhánh khi mọi lớp của nhánh đó có đủ train/validation và đạt precision, recall, accuracy ít nhất 85%; nếu chưa đạt, nhánh đó tiếp tục dùng nhận dạng heuristic để tránh nhãn neural đoán bừa. API tự nạp checkpoint hợp lệ; model không train lại mỗi lần chạy. Sau khi thêm ảnh mới, cần chạy lệnh train lại. Nếu train lỗi trước khi hoàn tất, checkpoint đích chưa được cập nhật.

Neural chỉ thay nhãn heuristic khi confidence đạt 0.85 và nhánh tương ứng vượt quality gate trong validation. Khi chưa cài PyTorch hoặc chưa có checkpoint đạt gate, API tiếp tục chạy nhận dạng hiện có. Nếu checkpoint không đạt yêu cầu, báo cáo cuối lệnh train chỉ rõ lớp còn thiếu mẫu hoặc độ chính xác thấp; hãy bổ sung/kiểm tra nhãn và train lại thay vì chỉ tăng epoch.

Checkpoint protocol cũ (trước v3) không được nạp vì bộ nhãn khối đã được tách `cube` và `cuboid`. Sau khi cập nhật mã, cần train lại để tạo checkpoint tương thích; nếu chưa đủ dữ liệu cho một nhánh, nhánh đó sẽ tiếp tục dùng heuristic.

### Đánh giá độ chính xác

Validation nội bộ giúp so sánh các lần train, nhưng cần tập test độc lập gồm ảnh/video thật từ các lần quay khác. Kiểm tra riêng độ chính xác 2D và 3D, từng lớp, cùng các trường hợp nhầm lẫn; không chỉ nhìn accuracy tổng. Mục tiêu 85% chỉ được xác nhận khi đạt trên tập video thật holdout đại diện cho điều kiện vận hành. Các ảnh render SolidWorks và dữ liệu tăng cường không thay thế đánh giá thực tế.

Kết quả hình dạng trong video được làm ổn định bằng đồng thuận qua nhiều frame; track tồn tại tối đa 20 frame bị mất trước khi được đóng để tránh ghép nhầm vật thể khác.

## Kiểm thử

```powershell
pytest -q --basetemp=.pytest-tmp
ruff check .
```

`tests/test_shape_benchmark.py` sinh ảnh 2D/3D với vị trí, kích thước, góc xoay, nền và độ sáng thay đổi; kiểm tra riêng phân loại đường viền và định vị detector. `tests/test_shape_learning.py` tách mẫu train/holdout theo seed để đo khả năng học từ kho mẫu. `tests/test_video_processor.py` kiểm tra frame lặp, video đầu ra và track khi vật thể di chuyển. Chạy lại toàn bộ suite sau mỗi thay đổi nhận dạng.

Kho mẫu dùng descriptor phiên bản 2 và được nâng cấp từ ảnh mẫu hiện có khi khởi động. Benchmark tổng hợp và ảnh train nội bộ không đại diện cho độ chính xác trên dây chuyền thật; cần đánh giá thêm bằng video thật đã gán nhãn, tách riêng khỏi ảnh huấn luyện.

## Hướng phát triển

1. Thay `ColorBlobDetector` bằng model detection đã huấn luyện.
2. Thêm adapter cho webcam, RTSP hoặc camera công nghiệp.
3. Bổ sung ROI, tiêu chí kiểm tra theo SKU và lưu kết quả vào database.
4. Thêm luồng video và giao diện biểu đồ năng suất/lỗi.
