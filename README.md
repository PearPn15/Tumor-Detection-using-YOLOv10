🧠 Đề tài: Phát hiện khối u não từ ảnh MRI sử dụng YOLOv10
🎯 Mục tiêu
Xây dựng một mô hình học sâu sử dụng kiến trúc YOLOv10 để tự động phát hiện và phân loại các loại khối u não từ ảnh chụp cộng hưởng từ (MRI), từ đó hỗ trợ chẩn đoán y tế một cách nhanh chóng và chính xác.

📂 Thông tin về bộ dữ liệu
📦 Nguồn: MRI for Brain Tumor with Bounding Boxes – Kaggle

📸 Tổng số ảnh: 3.906 ảnh MRI được chú thích bằng hộp giới hạn (bounding boxes) theo định dạng YOLO.

🏷️ Số lớp (4 lớp):

Lớp 0: U thần kinh đệm (Glioma)

Lớp 1: U màng não (Meningioma)

Lớp 2: Không có khối u (No Tumor)

Lớp 3: Tuyến yên (Pituitary tumor)

🔢 Phân tách dữ liệu
Train set
Loại u	Số lượng ảnh
U thần kinh đệm	1.153
U màng não	1.449
Không có khối u	711
Tuyến yên	1.424

Validation set
Loại u	Số lượng ảnh
U thần kinh đệm	136
U màng não	140
Không có khối u	100
Tuyến yên	136

