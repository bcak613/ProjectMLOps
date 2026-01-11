import os

# Nội dung file 1: Master Plan
master_plan_content = """# MLOps Master Plan: Customer Churn Prediction Project

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
"""

# Nội dung file 2: Concepts Guide
concepts_guide_content = """# MLOps Concepts Guide: Data Drift & Google MLOps Maturity

**Author:** AI Vietnam (MLOps Expert)
**Context:** Tài liệu bổ trợ lý thuyết cho dự án *Customer Churn Prediction*.

---

## 1. Data Drift là gì? (Tại sao Model hôm nay tốt, ngày mai lại tệ?)

Trong lập trình phần mềm truyền thống (ví dụ: web app), code bạn viết hôm nay chạy đúng thì 10 năm sau vẫn chạy đúng (nếu môi trường không đổi). Nhưng trong Machine Learning, **code không đổi nhưng kết quả vẫn có thể sai**. Đó là do dữ liệu thay đổi.

**Data Drift** là hiện tượng phân phối thống kê của dữ liệu thực tế (Live Data) thay đổi so với dữ liệu dùng để huấn luyện (Training Data), khiến hiệu suất mô hình suy giảm theo thời gian.

### Các loại Drift chính trong bài toán Churn Prediction:

#### A. Covariate Shift (Trôi dạt đầu vào)
* **Định nghĩa:** Phân phối của biến đầu vào (X) thay đổi, nhưng mối quan hệ giữa X và Y vẫn giữ nguyên.
* **Ví dụ:**
    * *Training:* Bạn train model với khách hàng chủ yếu 20-30 tuổi.
    * *Production:* Đột nhiên chiến dịch marketing thu hút toàn khách hàng 50-60 tuổi.
    * -> Model chưa từng "học" hành vi của nhóm 50-60 tuổi này, nên dự đoán sai.

#### B. Concept Drift (Trôi dạt khái niệm)
* **Định nghĩa:** Mối quan hệ giữa đầu vào (X) và nhãn dự đoán (Y) thay đổi. Đây là loại nguy hiểm nhất.
* **Ví dụ:**
    * *Trước đây:* Khách hàng "gọi điện > 100 phút/tháng" là khách hàng trung thành (Không rời bỏ).
    * *Hiện tại:* Đối thủ tung ra gói cước miễn phí gọi thoại. Bây giờ, khách hàng gọi nhiều vẫn rời bỏ mạng của bạn để sang đối thủ.
    * -> Quy luật cũ ("gọi nhiều = trung thành") đã sai. Model cũ trở nên vô dụng.

### Giải pháp trong dự án này:
Chúng ta sử dụng **Evidently AI** để giám sát:
1.  Thu thập log dữ liệu khi chạy API.
2.  So sánh phân phối (Distribution) của log này với dữ liệu gốc (Reference Data).
3.  Nếu phát hiện sai lệch lớn (Drift detected) -> Kích hoạt cảnh báo hoặc tự động Retrain model.

---

## 2. Kiến trúc MLOps Level 2 (Theo chuẩn Google)

Google chia độ trưởng thành của hệ thống MLOps thành 3 cấp độ (Level 0, 1, 2). Dự án chúng ta đang hướng tới **Level 2 - Cấp độ cao nhất**.

### Level 0: Quy trình thủ công (Manual Process)
* **Đặc điểm:** Data Scientist (DS) nhận dữ liệu, xử lý và train model trên Jupyter Notebook máy cá nhân. Khi có model, họ gửi file `.pkl` hoặc `.json` cho Dev để deploy.
* **Vấn đề:**
    * Khó tái lập (Reproducibility): "Code chạy trên máy tôi nhưng không chạy trên máy bạn".
    * Tách biệt giữa ML và Ops.
    * Không có Active Monitoring.

### Level 1: Tự động hóa Pipeline (ML Pipeline Automation)
* **Đặc điểm:** Tự động hóa quy trình training (CT - Continuous Training).
* **Cơ chế:** Khi có dữ liệu mới, hệ thống tự động kích hoạt pipeline: *Lấy dữ liệu -> Xử lý -> Train -> Validate -> Ra model mới*.
* **Vấn đề:** Mặc dù việc train tự động, nhưng việc triển khai code mới của pipeline (ví dụ: thay đổi thuật toán xử lý dữ liệu) vẫn làm thủ công.

### Level 2: Tự động hóa CI/CD (CI/CD Pipeline Automation)
Đây là đích đến của dự án này.

* **Định nghĩa:** Không chỉ tự động hóa việc train model (CT), mà tự động hóa cả việc **kiểm thử và triển khai chính cái Pipeline đó**.
* **Sự khác biệt cốt lõi:** Trong Level 2, "Sản phẩm" không phải là cái Model, mà là cái **Hệ thống tạo ra Model**.
* **Quy trình trong dự án của chúng ta:**
    1.  **CI (Continuous Integration):** Khi bạn sửa code `train.py` và push lên GitHub:
        * GitHub Actions chạy Unit Test.
        * Kiểm tra code style, kiểm tra tích hợp các module.
    2.  **CD (Continuous Deployment):**
        * Hệ thống tự động đóng gói code mới thành Docker Image.
        * Deploy phiên bản mới của API (FastAPI) lên môi trường Staging/Production.
    3.  **CT (Continuous Training):**
        * Hệ thống Monitoring (Evidently) phát hiện Drift -> Tự động trigger quy trình Train lại -> Ra model mới -> Tự động update vào API mà không cần Dev can thiệp.

### Tóm tắt sự khác biệt:
| Level | Code đổi thì sao? | Data đổi thì sao? | Thời gian deploy |
| :--- | :--- | :--- | :--- |
| **Level 0** | Làm thủ công | Train lại thủ công | Tuần/Tháng |
| **Level 1** | Deploy thủ công | **Tự động Train lại** | Ngày |
| **Level 2** | **Tự động Test & Deploy** | **Tự động Train lại** | **Phút/Giờ** |

---

> **Kết luận:** Việc bạn xây dựng pipeline với DVC, GitHub Actions (CI/CD), và MLflow chính là để đạt được **Level 2**: Một hệ thống khép kín, tự sửa chữa (self-healing) khi dữ liệu thay đổi và tự cập nhật khi code thay đổi.
"""

# Nội dung file 3: Execution Plan
execution_plan_content = """# Kế hoạch Triển khai Dự án: Customer Churn Prediction (MLOps Level 2)

**Role:** Team Lead
**Thời gian dự kiến:** 4 Tuần (4 Sprints)
**Mục tiêu:** Xây dựng hệ thống dự đoán Churn tự động hóa hoàn toàn từ Data -> Training -> Deploy.

---

## TUẦN 1: FOUNDATION & DATA PIPELINE (Xây móng)
**Mục tiêu:** Thiết lập hạ tầng và luồng dữ liệu sạch vào Feature Store.

### 1.1. Hạ tầng (Infrastructure) - *Người phụ trách: DevOps/MLOps*
* [ ] **Repository Setup:** Tạo GitHub Repo, cấu trúc thư mục chuẩn (`src`, `data`, `notebooks`, `.github`, `scripts`).
* [ ] **Environment:** Tạo file `environment.yml` (Conda) và `requirements.txt`. Đảm bảo team dùng chung phiên bản Python (3.9 hoặc 3.10).
* [ ] **Service Setup:** Dựng Docker Compose cho các dịch vụ nền tảng:
    * **MinIO:** Giả lập S3 để lưu trữ Data & Artifacts.
    * **PostgreSQL:** Backend cho MLflow và Feast.
    * **Redis:** Online Store cho Feast.
    * **MLflow Server:** Dashboard theo dõi thí nghiệm.

### 1.2. Data Engineering - *Người phụ trách: Data Engineer*
* [ ] **DVC Initialization:** Cài đặt DVC, cấu hình remote storage trỏ về MinIO.
* [ ] **Data Versioning:** Thực hiện `dvc add data/raw/churn.csv` và push lên MinIO.
* [ ] **ETL Script:** Viết script `process_data.py`:
    * Clean dữ liệu.
    * Split Train/Test.
    * Lưu output dưới dạng **Parquet** (để tối ưu cho Feast).

### 1.3. Feature Store - *Người phụ trách: Data Engineer + Data Scientist*
* [ ] **Feast Definitions:** Định nghĩa file `feature_store.yaml` và `definitions.py` (Entity, Feature Views).
* [ ] **Materialization:** Chạy lệnh `feast materialize` để đẩy dữ liệu từ Parquet (Offline) lên Redis (Online).
* [ ] **Test:** Viết script nhỏ `test_feast.py` để thử query một feature vector từ Redis xem tốc độ có < 10ms không.

---

## TUẦN 2: MODEL PIPELINE & TRACKING (Dựng khung nhà)
**Mục tiêu:** Có được mô hình tốt nhất và quản lý được các phiên bản thí nghiệm.

### 2.1. Experimentation - *Người phụ trách: Data Scientist*
* [ ] **Baseline Model:** Train model XGBoost cơ bản trên Notebook để làm mốc so sánh.
* [ ] **Refactor Code:** Chuyển code từ Notebook sang script `src/train.py`.
* [ ] **MLflow Integration:** Gắn `mlflow.xgboost.autolog()` vào code training.
* [ ] **Custom Logging:** Log thêm các metrics quan trọng: F1-Score, AUC. Log `confusion_matrix.png` và `shap_summary.png` dưới dạng Artifacts.

### 2.2. Model Registry - *Người phụ trách: ML Engineer*
* [ ] **Registry Workflow:** Thiết lập quy trình đăng ký model.
    * Model tốt nhất sẽ được register với tên `churn-prediction-model`.
    * Sử dụng Alias: `@Staging` cho model vừa train xong, `@Champion` cho model đang chạy Production.
* [ ] **Evaluation Script:** Viết `eval.py` để load model và test trên tập dữ liệu kiểm thử, đảm bảo metrics đạt ngưỡng (threshold) đề ra.

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
"""

files = {
    "Project_MasterPlan_MLOps_Churn.md": master_plan_content,
    "MLOps_Concepts_Guide.md": concepts_guide_content,
    "Project_Execution_Plan.md": execution_plan_content
}

def create_files():
    print("🚀 Đang khởi tạo tài liệu dự án MLOps...")
    for filename, content in files.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Đã tạo file: {filename}")
    print("\n🎉 Hoàn tất! Bạn đã sẵn sàng để gửi tài liệu cho team.")

if __name__ == "__main__":
    create_files()