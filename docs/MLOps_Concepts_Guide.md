# MLOps Concepts Guide: Data Drift & Google MLOps Maturity

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
