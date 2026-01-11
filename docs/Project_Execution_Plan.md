# Kế hoạch Triển khai Dự án: Customer Churn Prediction (MLOps Level 2)

**Role:** Team Lead
**Thời gian dự kiến:** 4 Tuần (4 Sprints)
**Mục tiêu:** Xây dựng hệ thống dự đoán Churn tự động hóa hoàn toàn từ Data -> Training -> Deploy.

---

## TUẦN 1: FOUNDATION & DATA PIPELINE (Xây móng)
**Mục tiêu:** Thiết lập hạ tầng và luồng dữ liệu sạch vào Feature Store.

### 1.1. Hạ tầng (Infrastructure) - *Người phụ trách: DevOps/MLOps*
* [x] **Repository Setup:** Tạo GitHub Repo, cấu trúc thư mục chuẩn (`src`, `data`, `notebooks`, `.github`, `scripts`).
* [x] **Environment:** Tạo file `environment.yml` (Conda) và `requirements.txt`. Đảm bảo team dùng chung phiên bản Python (3.9 hoặc 3.10).
* [x] **Service Setup:** Dựng Docker Compose cho các dịch vụ nền tảng:
    * **MinIO:** Giả lập S3 để lưu trữ Data & Artifacts.
    * **PostgreSQL:** Backend cho MLflow và Feast.
    * **Redis:** Online Store cho Feast.
    * **MLflow Server:** Dashboard theo dõi thí nghiệm.

### 1.2. Data Engineering - *Người phụ trách: Data Engineer*
* [x] **DVC Initialization:** Cài đặt DVC, cấu hình remote storage trỏ về MinIO.
* [x] **Data Versioning:** Thực hiện `dvc add data/raw/churn.csv` và push lên MinIO.
* [x] **ETL Script:** Viết script `process_data.py`:
    * Clean dữ liệu.
    * Split Train/Test.
    * Lưu output dưới dạng **Parquet** (để tối ưu cho Feast).

### 1.3. Feature Store - *Người phụ trách: Data Engineer + Data Scientist*
* [x] **Feast Definitions:** Định nghĩa file `feature_store.yaml` và `definitions.py` (Entity, Feature Views).
* [x] **Materialization:** Chạy lệnh `feast materialize` để đẩy dữ liệu từ Parquet (Offline) lên Redis (Online).
* [x] **Test:** Viết script nhỏ `test_feast.py` để thử query một feature vector từ Redis xem tốc độ có < 10ms không.

---

## TUẦN 2: MODEL PIPELINE & TRACKING (Dựng khung nhà)
**Mục tiêu:** Có được mô hình tốt nhất và quản lý được các phiên bản thí nghiệm.

### 2.1. Experimentation - *Người phụ trách: Data Scientist*
* [x] **Baseline Model:** Train model XGBoost cơ bản trên Notebook để làm mốc so sánh.
* [x] **Refactor Code:** Chuyển code từ Notebook sang script `src/train.py`.
* [x] **MLflow Integration:** Gắn `mlflow.xgboost.autolog()` vào code training.
* [x] **Custom Logging:** Log thêm các metrics quan trọng: F1-Score, AUC. Log `confusion_matrix.png` và `shap_summary.png` dưới dạng Artifacts.

### 2.2. Model Registry - *Người phụ trách: ML Engineer*
* [x] **Registry Workflow:** Thiết lập quy trình đăng ký model.
    * Model tốt nhất sẽ được register với tên `churn-prediction-model`.
    * Sử dụng Alias: `@Staging` cho model vừa train xong, `@Champion` cho model đang chạy Production.
* [x] **Evaluation Script:** Viết `eval.py` để load model và test trên tập dữ liệu kiểm thử, đảm bảo metrics đạt ngưỡng (threshold) đề ra.

---

## TUẦN 3: CI/CD AUTOMATION (Lắp dây chuyền sản xuất)
**Mục tiêu:** Mọi thao tác push code đều kích hoạt quy trình tự động (Level 2 Requirement).

### 3.1. GitHub Actions Runner - *Người phụ trách: DevOps*
* [ ] **Self-hosted Runner:** Cài đặt Runner trên máy server (hoặc máy local mạnh) để chạy pipeline nhanh hơn GitHub Cloud free tier.
* [ ] **Connect:** Kết nối Runner với Repo qua Token.

### 3.2. CI Pipeline (Continuous Integration) - *Người phụ trách: MLOps*
* [ ] **Workflow `training.yaml`:**
    * Trigger: Khi push vào nhánh `main` hoặc `dev`.
    * Steps: Pull Data (DVC) -> Setup Env -> Run `train.py` -> Run `eval.py`.
* [ ] **Auto-Promotion Logic:** Nếu `eval.py` trả về kết quả tốt hơn model hiện tại -> Tự động gắn tag `@Champion` cho model mới (hoặc gửi Alert cho Lead duyệt).

### 3.3. CD Pipeline (Continuous Deployment) - *Người phụ trách: MLOps*
* [ ] **Containerization:** Viết `Dockerfile` cho API Service.
* [ ] **Workflow `deploy.yaml`:**
    * Build Docker Image.
    * Restart container API với image mới nhất.

---

## TUẦN 4: SERVING & MONITORING (Hoàn thiện & Bàn giao)
**Mục tiêu:** Đưa model ra phục vụ người dùng và giám sát sức khỏe hệ thống.

### 4.1. Serving API - *Người phụ trách: ML Engineer*
* [ ] **FastAPI App:** Viết API `/predict`:
    * Input: `customer_id`.
    * Logic: Lấy feature từ Feast (Redis) -> Predict qua Model (Load từ MLflow).
    * Output: `churn_probability`.
* [ ] **Gradio Interface:** Tạo giao diện demo đơn giản để team Business dùng thử.

### 4.2. Monitoring - *Người phụ trách: Data Scientist/MLOps*
* [ ] **Evidently AI Setup:**
    * Tạo Reference Dataset (dữ liệu lúc train).
    * Thu thập Current Dataset (log từ API).
* [ ] **Drift Report:** Tạo Dashboard HTML hiển thị Data Drift và Model Drift.
* [ ] **Alert:** Cấu hình cảnh báo nếu phát hiện Drift nghiêm trọng.

---

## RỦI RO & GIẢI PHÁP (Risk Management)

| Rủi ro | Mức độ | Giải pháp |
| :--- | :--- | :--- |
| **Môi trường không đồng nhất** | Cao | Bắt buộc dùng Docker/Conda ngay từ ngày 1. Cấm dùng `pip install` thẳng vào máy local. |
| **Dữ liệu training quá lớn** | Trung bình | Sử dụng DVC với `.dvcignore` cẩn thận, chỉ pull về mẫu nhỏ (sample) để test pipeline CI/CD trước. |
| **Feast cấu hình sai** | Cao | Feast rất hay lỗi kết nối Redis. Cần test kỹ kết nối mạng giữa các container trong Docker Compose. |
| **Model mới tệ hơn model cũ** | Thấp | Luôn giữ lại model `@Champion` cũ. Pipeline CI/CD phải có bước so sánh metrics trước khi Promote. |

---

## ĐỊNH NGHĨA HOÀN THÀNH (Definition of Done - DoD)

1.  Code được push lên GitHub đầy đủ, sạch sẽ.
2.  Pipeline chạy xanh (Green tick) trên GitHub Actions.
3.  API phản hồi dưới 200ms.
4.  Dashboard MLflow hiển thị đầy đủ thông số training.
5.  Dashboard Evidently hiển thị được báo cáo drift.
