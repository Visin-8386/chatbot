# 🚀 Hướng dẫn Deployment Dự án DocSearch

Dự án **DocSearch** là một hệ thống RAG (Retrieval-Augmented Generation) chạy hoàn toàn offline. Để triển khai (deploy) dự án này một cách chuyên nghiệp, bạn có 3 lựa chọn chính tùy thuộc vào môi trường của bạn.

---

## 📋 Yêu Cầu Hệ Thống (Khuyến nghị)
Vì dự án sử dụng LLM (Qwen 2.5) và Embedding models, để có tốc độ phản hồi tốt nhất:
- **Hệ điều hành**: Linux (Ubuntu 22.04+) hoặc Windows 10/11.
- **GPU**: NVIDIA (RTX 3060 trở lên, tối thiểu 6GB VRAM) - **Cực kỳ quan trọng để chạy mượt**.
- **RAM**: Tối thiểu 16GB.
- **Docker & Docker Compose**: Nếu bạn muốn triển khai bằng container.

---

## 🛠 Lựa chọn 1: Triển khai bằng Docker Compose (Khuyên dùng)

Đây là cách tốt nhất để triển khai lên Server vì nó tự động cấu hình môi trường và hỗ trợ GPU NVIDIA một cách chuẩn xác.

### 1. Cài đặt NVIDIA Container Toolkit
Nếu bạn dùng Linux, bạn cần cài đặt toolkit này để Docker có thể sử dụng GPU:
```bash
# Ví dụ trên Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# ... (làm theo hướng dẫn chính thức của NVIDIA)
```

### 2. Cấu hình file `.env`
Đảm bảo bạn đã có file `.env` trong thư mục gốc. Bạn có thể copy từ `.env.example`:
```bash
cp .env.example .env
```

### 3. Khởi chạy
Chỉ cần một câu lệnh duy nhất:
```bash
docker-compose up -d --build
```
*Hệ thống sẽ tự động build image, cài đặt thư viện và mount các thư mục `uploads`, `data`, `logs` ra máy host để dữ liệu không bị mất khi restart container.*

---

## 🐍 Lựa chọn 2: Triển khai Thủ công (Manual)

Phù hợp nếu bạn muốn chạy trực tiếp trên máy tính cá nhân hoặc Server không dùng Docker.

### 1. Tạo môi trường ảo (Virtual Env)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows
```

### 2. Cài đặt thư viện
```bash
pip install -r backend/requirements.txt
```

### 3. Chạy Server Production với Uvicorn
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```
*Lưu ý: Không nên dùng `--reload` khi deploy production. Số lượng `workers` nên để là 1 vì mô hình LLM chiếm dụng VRAM rất lớn, chạy nhiều worker dễ gây tràn bộ nhớ GPU.*

---

## ☁️ Lựa chọn 3: Triển khai lên Cloud (RunPod, Lambda Labs, AWS)

Nếu bạn không có GPU cá nhân, bạn có thể thuê GPU theo giờ:

1.  **RunPod / Lambda Labs**: Thuê một instance có GPU (ví dụ RTX 4090 hoặc A10G).
2.  **SSH vào server**: Clone code từ GitHub.
3.  **Deploy bằng Docker**: Làm theo bước ở **Lựa chọn 1**.
4.  **Mở Port**: Đảm bảo mở port `8000` trên Firewall của nhà cung cấp Cloud.

---

## 🛡️ Các lưu ý quan trọng khi Production

1.  **API Key**: Trong file `.env`, hãy đặt giá trị cho `API_KEY` để bảo vệ các endpoint tải lên và xóa tài liệu.
2.  **Preload Model**: Mặc định dự án sẽ tải model khi startup (`startup_preload`). Việc này khiến server khởi động lâu (~30s-1p) nhưng giúp query đầu tiên của người dùng phản hồi ngay lập tức.
3.  **Dữ liệu**: Thư mục `data/` chứa ChromaDB (Vector Store). Đừng quên backup thư mục này thường xuyên.
4.  **HTTPS**: Nếu deploy lên internet, bạn nên sử dụng **Nginx** làm Reverse Proxy và cấu hình **SSL (Let's Encrypt)** để bảo mật dữ liệu truyền tải.

---

## 🔍 Kiểm tra sau khi Deploy
Sau khi chạy thành công, hãy truy cập:
- Giao diện người dùng: `http://<your-ip>:8000`
- API Docs: `http://<your-ip>:8000/docs`
- Kiểm tra sức khỏe hệ thống: `http://<your-ip>:8000/api/health`

Chúc bạn deploy thành công! 🚀
