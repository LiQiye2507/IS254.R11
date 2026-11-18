# Phân tích kiểm chứng giả thuyết H1, H2, H3

Số liệu lấy từ kiểm định chéo 5-fold (`StratifiedGroupKFold` theo `id_student`) trong [experimental_results.md](experimental_results.md). Mẫu đánh giá gồm **29.915 lượt đăng ký** (26.740 sinh viên, tỷ lệ At-risk 48.57%): toàn bộ lượt đăng ký trong `studentInfo.csv`, trừ sinh viên hủy môn trước ngày bắt đầu khóa học. Mẫu này bao gồm cả 4.101 lượt chưa nộp bài nào; trong nhóm này 99.9% là Withdrawn hoặc Fail.

Các so sánh giữa hai mô hình dùng **hiệu số theo từng fold** (mục 4 của file kết quả): hai mô hình được đánh giá trên cùng một tập test, nên hiệu số phản ánh chênh lệch thật giữa mô hình, không lẫn với biến động do cách chia dữ liệu.

> **Lưu ý về ngưỡng:** DBN và LSTM chọn ngưỡng phân loại tối ưu F1 trên tập validation; mô hình đối chứng (RF, LR) dùng ngưỡng cố định 0.5. Accuracy, Precision, Recall và F1 vì vậy phụ thuộc cả vào cách chọn ngưỡng. **AUC-ROC và AUC-PR không phụ thuộc ngưỡng**, nên là thước đo chính khi so sánh chất lượng mô hình.

## Tóm tắt

| Giả thuyết | Kết luận | Căn cứ chính |
| :-- | :-- | :-- |
| **H1** – Đầu kỳ (T ≤ 2), DBN ổn định hơn hoặc tương đương LSTM | **Ủng hộ một phần** | F1 tương đương, dao động F1/Recall giữa các fold nhỏ hơn; nhưng AUC-ROC thấp hơn LSTM ở 5/5 fold và kém ổn định hơn về AUC |
| **H2** – Từ T ≥ 4, LSTM có F1 và AUC-ROC cao hơn DBN | **Được ủng hộ** (so với DBN) | AUC-ROC cao hơn ở 5/5 fold tại mọi mốc T ≥ 4; F1 cao hơn ở 5/5 fold, trừ T = 16 (tương đương) |
| ↳ Lý do "nhờ học được chuỗi hành vi" | **Chưa được ủng hộ** | Random Forest chỉ dùng đặc trưng tích lũy đến T đạt AUC-ROC ngang LSTM ở mọi mốc tuần |
| **H3** – DBN cung cấp giải thích các yếu tố ảnh hưởng | **Được ủng hộ** (định tính) | DBN trả về ΔP cho từng yếu tố của từng sinh viên; LSTM không có sẵn khả năng này |

---

## H1. Đầu kỳ (T = 2): DBN so với LSTM

| Chỉ số | DBN | LSTM | DBN − LSTM (theo fold) | Số fold DBN > LSTM |
| :-- | :-: | :-: | :-: | :-: |
| AUC-ROC | 0.6271 ± 0.0046 | 0.6744 ± 0.0030 | −0.0473 ± 0.0044 | 0/5 |
| AUC-PR | 0.5964 ± 0.0081 | 0.6511 ± 0.0028 | – | – |
| F1 | 0.6543 ± 0.0065 | 0.6519 ± 0.0116 | +0.0025 ± 0.0089 | 3/5 |
| Recall | 0.8515 ± 0.0259 | 0.7432 ± 0.0398 | +0.1083 ± 0.0476 | 5/5 |
| Precision | 0.5316 ± 0.0078 | 0.5815 ± 0.0088 | – | – |
| Accuracy | 0.5631 ± 0.0103 | 0.6150 ± 0.0056 | – | – |

**Phần được ủng hộ:**
- **Tương đương về F1:** chênh lệch +0.0025 nhỏ hơn nhiều so với độ lệch chuẩn của chính nó (0.0089), và DBN chỉ cao hơn ở 3/5 fold.
- **Ổn định hơn ở các chỉ số theo ngưỡng:** độ lệch chuẩn giữa các fold của DBN nhỏ hơn LSTM ở F1 (0.0065 so với 0.0116), Recall (0.0259 so với 0.0398) và Precision.

**Phần không được ủng hộ:**
- **Khả năng xếp hạng nguy cơ:** AUC-ROC của DBN thấp hơn LSTM ở cả 5 fold (−0.047), AUC-PR cũng thấp hơn.
- **Không ổn định hơn ở mọi chỉ số:** trên các chỉ số không phụ thuộc ngưỡng, DBN lại dao động nhiều hơn LSTM (AUC-ROC 0.0046 so với 0.0030; AUC-PR 0.0081 so với 0.0028), và Accuracy cũng vậy.
- **Recall cao hơn không có nghĩa là mô hình tốt hơn:** Recall 0.85 của DBN đi kèm Precision thấp nhất (0.53) và Accuracy thấp nhất (0.56). DBN đang cảnh báo rộng tay hơn, đây là đánh đổi ở bước chọn ngưỡng, không phải ưu thế phân biệt của mô hình.
- **DBN không phải mô hình tốt nhất ở T = 2:** cả hai mô hình đối chứng đều có AUC-ROC cao hơn (LR 0.6793, RF 0.6786).

**Nguyên nhân có thể (cần kiểm chứng thêm):**
1. **Đến hết tuần 2 gần như chưa có dữ liệu học tập.** Chỉ **4.5%** lượt đăng ký có bài đánh giá đến hạn và 12.8% có bất kỳ hoạt động nộp bài nào. Ở T = 2, mọi mô hình về thực chất chỉ dự đoán từ đặc trưng nền (nhân khẩu học, học vấn, môn học).
2. **Rời rạc hóa làm DBN mất thông tin nền.** DBN dùng các nhóm thô (IMD 4 mức, tín chỉ 3 mức, số lần học lại 3 mức), trong khi LR/RF dùng one-hot đầy đủ và giá trị số liên tục. Khi chỉ còn đặc trưng nền, mất mát này thể hiện rõ nhất.
3. **Pipeline chưa dùng dữ liệu click LMS (`studentVle`).** H1 phát biểu cho tình huống "dữ liệu LMS còn ít", nhưng hệ thống hiện chưa đưa click vào mô hình nào. Vì vậy H1 mới chỉ được kiểm chứng trên dữ liệu bài đánh giá, chưa đúng như phát biểu ban đầu.

---

## H2. Từ giữa kỳ (T ≥ 4): LSTM so với DBN

| Tuần | ΔAUC-ROC (LSTM − DBN) | Số fold LSTM > DBN | ΔF1 (LSTM − DBN) | Số fold LSTM > DBN |
| :-: | :-: | :-: | :-: | :-: |
| 4 | +0.0417 | 5/5 | +0.0249 | 5/5 |
| 8 | +0.0355 | 5/5 | +0.0171 | 5/5 |
| 12 | +0.0331 | 5/5 | +0.0193 | 5/5 |
| 16 | +0.0182 | 5/5 | +0.0027 | 2/5 |
| 20 | +0.0155 | 5/5 | +0.0102 | 5/5 |

**H2 được ủng hộ:** AUC-ROC của LSTM vượt DBN nhất quán ở mọi mốc tuần, mọi fold. Về F1, LSTM cũng vượt ở mọi mốc trừ T = 16, nơi hai mô hình tương đương. Khoảng cách AUC-ROC lớn nhất ở T = 4 rồi thu hẹp dần về cuối kỳ, khi dữ liệu tích lũy đủ nhiều để cả hai mô hình cùng phân biệt tốt.

**Tuy nhiên, lý do mà H2 nêu ("nhờ học được chuỗi hành vi phức tạp") chưa được ủng hộ:**

| Tuần | ΔAUC-ROC (LSTM − RF) | Số fold LSTM > RF | ΔAUC-ROC (LSTM − LR) | Số fold LSTM > LR |
| :-: | :-: | :-: | :-: | :-: |
| 4 | −0.0043 | 0/5 | +0.0069 | 5/5 |
| 8 | −0.0014 | 1/5 | +0.0038 | 5/5 |
| 12 | +0.0005 | 3/5 | +0.0071 | 5/5 |
| 16 | +0.0002 | 3/5 | +0.0077 | 5/5 |
| 20 | −0.0013 | 1/5 | +0.0041 | 5/5 |

- **RF ngang LSTM về AUC-ROC:** Random Forest chỉ nhìn bảng đặc trưng tích lũy đến tuần T, không có thông tin thứ tự thời gian, nhưng đạt AUC-ROC ngang LSTM; mọi chênh lệch đều dưới 0.005.
- **F1 và Recall cao hơn của LSTM chưa so được:** LSTM có F1 cao hơn RF ở T ≤ 12 (5/5 fold, +0.006 đến +0.035), nhưng chủ yếu do khác biệt về ngưỡng. RF dùng ngưỡng cố định 0.5 nên Precision cao, Recall thấp; LSTM dùng ngưỡng tối ưu F1. Muốn kết luận về F1 cần chọn ngưỡng cho RF theo cùng cách.
- **LSTM chỉ vượt Logistic Regression rất ít:** chỉ +0.004 đến +0.008 AUC-ROC, tức là lợi ích chủ yếu đến từ việc mô hình hóa phi tuyến, mà RF cũng làm được.

**Nguyên nhân có thể:**
1. **Chuỗi đầu vào của LSTM đã chứa sẵn đặc trưng tích lũy.** 10/15 kênh trong tensor tuần là đại lượng tích lũy đến tuần t (số bài đã nộp, số bài bỏ, điểm trung bình trọng số…), giống thông tin RF nhận được. Phần "chuỗi" thực sự bổ sung chỉ là thời điểm xảy ra sự kiện.
2. **Chuỗi tuần rất thưa.** Đến tuần 20, trung bình mỗi lượt đăng ký chỉ có 4.4 bài đánh giá định kỳ đến hạn; phần lớn các tuần không có sự kiện nào. Dữ liệu click LMS hằng tuần mới là loại chuỗi dày mà LSTM có lợi thế, nhưng hiện chưa được sử dụng.

**Ý nghĩa với thiết kế DSS:** việc chọn LSTM làm mô hình dự báo chính nên được lập luận bằng **lợi thế vận hành**, không phải độ chính xác:
- Một mô hình duy nhất cho xác suất ở mọi tuần 1..20, nên vẽ được đường xu hướng rủi ro cho từng sinh viên.
- Baseline cần huấn luyện riêng một mô hình cho từng mốc tuần.

---

## H3. Khả năng giải thích của DBN

DBN trả về độ đóng góp ΔP của từng yếu tố cho từng sinh viên tại tuần T. Đây chính là thông tin hiển thị ở tab "Chẩn đoán & Giải thích" của dashboard. Ví dụ sinh viên `1414443` (môn AAA, 2013J) tại T = 8: P(At-risk) = 73.6%, yếu tố tăng rủi ro mạnh nhất là điểm tuần 5–8 dưới 40 (`perf_S3 = F_lt40`, +5.6%), tiếp theo là nhóm IMD thấp (+2.6%). Kết quả thực tế của sinh viên này là Fail.

**Giới hạn cần nêu trong báo cáo:**
- **ΔP là giải thích theo mô hình, không phải tác động nhân quả theo nghĩa can thiệp.** ΔP đo mức thay đổi xác suất khi bỏ một bằng chứng khỏi truy vấn. Cấu trúc mạng được học từ dữ liệu quan sát, nên không khẳng định được rằng thay đổi yếu tố đó sẽ thay đổi kết quả.
- **ΔP nhỏ không có nghĩa là yếu tố không quan trọng.** ΔP đo đóng góp của một yếu tố *khi các bằng chứng khác vẫn giữ nguyên*. Nếu hai yếu tố mang thông tin trùng nhau, ví dụ điểm của các lát cắt liên tiếp, bỏ một yếu tố thì yếu tố còn lại bù vào, nên ΔP của cả hai đều nhỏ.
- **H3 mới được kiểm chứng định tính.** Chưa có đánh giá với người dùng (cố vấn học tập) về việc giải thích có hữu ích cho quyết định can thiệp hay không.

---

## Hệ quả với mô hình Hybrid (RQ3)

- **Trọng số đầu kỳ chưa có căn cứ.** Hybrid trong dashboard đặt trọng số DBN 60% khi T ≤ 2, dựa trên giả định của H1. Kết quả trên cho thấy ở T = 2, DBN có AUC-ROC thấp nhất trong 4 mô hình, nên trọng số này chưa có căn cứ thực nghiệm.
- **Hybrid chưa được đánh giá.** Mô hình Hybrid chưa nằm trong benchmark, nên RQ3 về hiệu năng dự báo vẫn còn mở.

## Giới hạn của phân tích

- **5 fold chưa phải kiểm định thống kê chính thức.** Kết quả "5/5 fold" là bằng chứng nhất quán mạnh nhưng không thay được kiểm định đó; các chênh lệch nhỏ (dưới 0.005 AUC-ROC) nên xem là tương đương.
- **Cách chọn ngưỡng khác nhau giữa các mô hình.** Mô hình đối chứng dùng ngưỡng 0.5, DBN/LSTM dùng ngưỡng tối ưu trên validation, nên các so sánh F1/Recall giữa hai nhóm chưa hoàn toàn công bằng.
- **Kết quả gắn với một cấu hình cụ thể.** Mỗi mô hình dùng một bộ siêu tham số cố định, và đặc trưng chỉ lấy từ bài đánh giá; kết luận có thể thay đổi khi bổ sung dữ liệu click LMS.
- **Không so trực tiếp với số liệu của phiên bản trước.** Phiên bản trước chỉ đánh giá 25.814 lượt có ít nhất một bài nộp. Mẫu hiện tại thêm nhóm chưa nộp bài, vốn dễ nhận diện là At-risk từ giữa kỳ, nên các chỉ số tuyệt đối không so sánh trực tiếp được với phiên bản đó.
