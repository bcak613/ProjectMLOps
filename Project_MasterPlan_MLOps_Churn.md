# MLOps Master Plan: Customer Churn Prediction Project

**Author:** AI Vietnam (Synthesized by MLOps Expert)
**Project Goal:** Dự đoán khách hàng rời bỏ (Churn Prediction) với quy trình MLOps tự động hóa hoàn chỉnh.

---

## 1. Tech Stack Overview (Kiến trúc công nghệ)

* **Source Code & CI/CD:** GitHub, GitHub Actions (Self-hosted Runners).
* **Data Ops:** DVC (Data Version Control), MinIO/S3 (Object Storage).
* **Feature Store:** Feast, Redis (Online Store), Parquet (Offline Store).
* **Model Ops:** MLflow (Tracking & Registry), XGBoost (Model).
* **Serving:** FastAPI (Backend), Gradio (Frontend/Demo).
* **Monitoring:** Evidently AI.
* **Infrastructure:** Docker, Conda.

---

## 2. Step-by-Step Implementation Guide

### PHASE 1: DATA PIPELINE (Xây dựng luồng dữ liệu)
*Mục tiêu: Đảm bảo tính tái lập của dữ liệu (Reproducibility) và phục vụ Feature độ trễ thấp.*

**Bước 1.1: Quản lý phiên bản dữ liệu với DVC**
* **Action:**
    * Khởi tạo DVC trong dự án (`dvc init`).
    * Cấu hình remote storage (S3 bucket hoặc MinIO).
    * Thực hiện `dvc track` file dữ liệu gốc (CSV từ Kaggle).
    * Đẩy dữ liệu lên remote storage (`dvc push`).
* **Expert Note:** Luôn tách biệt code (Git) và data (DVC). Git chỉ lưu file `.dvc` hash nhẹ.

**Bước 1.2: Data Processing & Feature Engineering**
* **Action:**
    * Viết script xử lý làm sạch dữ liệu (Clean).
    * Chia tập dữ liệu (Split) thành Train/Test.
    * Chuyển đổi dữ liệu sang định dạng **Parquet** (tối ưu cho Feast).

**Bước 1.3: Triển khai Feature Store (Feast)**
* **Action:**
    * Định nghĩa `Feast Entity` (ví dụ: `customer_id`).
    * Định nghĩa `Feast View`: Liên kết với file Parquet đã xử lý.
    * Thực hiện `feast apply` để đăng ký Feature Registry.
    * **Materialize:** Đồng bộ feature từ Offline Store (Parquet) sang Online Store (Redis) để phục vụ Real-time Inference.
    * Lệnh: `feast materialize-incremental $(date +%Y-%m-%d)`

---

### PHASE 2: MODEL PIPELINE (Phát triển & Quản lý mô hình)
*Mục tiêu: Theo dõi thí nghiệm và quản lý vòng đời mô hình.*

**Bước 2.1: Experiment Tracking với MLflow**
* **Action:**
    * Dựng MLflow Server (kết nối với MinIO để lưu Artifacts và Postgres/SQLite để lưu Metadata).
    * Viết script training (`train.py`) sử dụng **XGBoost**.
    * Tích hợp `mlflow.xgboost.autolog()` hoặc log thủ công:
        * **Parameters:** Learning rate, max_depth, n_estimators...
        * **Metrics:** Accuracy, F1-Score, AUC, RMSE.
        * **Artifacts:** Model file, Confusion Matrix plot, Feature Importance plot.

**Bước 2.2: Model Evaluation & Explainability**
* **Action:**
    * Tính toán các chỉ số trên tập Test.
    * Sử dụng **SHAP** để giải thích mô hình (Feature Impact) - trả lời câu hỏi "Tại sao khách hàng này rời bỏ?".
    * Log các biểu đồ SHAP lên MLflow Artifacts.

**Bước 2.3: Model Registry (Đăng ký & Phân loại)**
* **Action:**
    * Đăng ký model tốt nhất vào **MLflow Model Registry**.
    * Gắn Alias (Nhãn):
        * `@Staging`: Phiên bản đang thử nghiệm.
        * `@Champion`: Phiên bản tốt nhất, sẵn sàng cho Production.
    * Quy trình Promotion: So sánh metrics của model mới với model `@Champion` hiện tại. Nếu tốt hơn -> Promote.

---

### PHASE 3: SERVING PIPELINE (Triển khai dịch vụ)
*Mục tiêu: Cung cấp API dự đoán với độ trễ thấp và giám sát độ trôi dữ liệu.*

**Bước 3.1: Build Prediction Service (FastAPI)**
* **Action:**
    * Load model từ MLflow Registry (sử dụng Alias `@Champion` để luôn lấy bản tốt nhất).
    * Kết nối với Feast Online Store.
    * **Luồng xử lý API `/predict`:**
        1.  Nhận `customer_id` từ Request.
        2.  Gọi `get_online_features()` từ Feast để lấy features mới nhất từ Redis.
        3.  Đưa features vào Model để dự đoán.
        4.  Trả về kết quả (Churn/No Churn).

**Bước 3.2: User Interface (Gradio)**
* **Action:**
    * Xây dựng giao diện đơn giản cho người dùng cuối nhập ID hoặc thông tin để test nhanh API.

**Bước 3.3: Monitoring (Evidently AI)**
* **Action:**
    * Thu thập dữ liệu thực tế (Inference data).
    * So sánh với dữ liệu huấn luyện (Reference data).
    * Phát hiện **Data Drift** (sự thay đổi phân phối dữ liệu) và **Concept Drift** (mô hình bị lỗi thời).
    * Cảnh báo nếu độ chính xác giảm.

---

### PHASE 4: CI/CD PIPELINE (Tự động hóa)
*Mục tiêu: "Code push -> Train -> Deploy" không cần can thiệp thủ công.*

**Bước 4.1: Setup Self-hosted Runner**
* **Action:**
    * Cấu hình máy server (Linux/EC2) làm GitHub Actions Runner (để tận dụng GPU hoặc môi trường cài sẵn).
    * Kết nối Runner với Repository thông qua Token.

**Bước 4.2: Define Workflows (`.github/workflows`)**
* **Trigger:** Khi có sự kiện `push` vào nhánh `main`.
* **Jobs:**
    1.  **Train Model:**
        * Checkout code.
        * Pull data từ DVC.
        * Chạy `train.py`.
        * So sánh kết quả, nếu tốt hơn thì cập nhật `@Champion` trong MLflow.
    2.  **Build Docker:** Đóng gói ứng dụng API.
    3.  **Deploy:** Restart service FastAPI với model mới nhất.

---

## 3. Checklist cho Team (Phân chia công việc)

| Role | Nhiệm vụ chính | Công cụ |
| :--- | :--- | :--- |
| **Data Engineer** | DVC setup, ETL script, Feast definition, Redis setup. | DVC, S3, Feast, SQL/Pandas |
| **Data Scientist** | Feature Engineering, XGBoost modeling, Hyperparameter tuning, SHAP analysis. | Jupyter, Scikit-learn, XGBoost |
| **ML Engineer** | MLflow setup, Model Registry workflow, FastAPI implementation, Dockerize. | MLflow, FastAPI, Docker |
| **DevOps/MLOps** | CI/CD pipeline (GitHub Actions), Monitoring (Evidently), Infrastructure. | GitHub Actions, Bash, Cloud (AWS/GCP) |
