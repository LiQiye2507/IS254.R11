# Phân tích kiểm chứng giả thuyết H1, H2, H3

Số liệu lấy từ kiểm định chéo 5-fold (`StratifiedGroupKFold` theo `id_student`) trong [experimental_results.md](experimental_results.md). Các so sánh giữa hai mô hình dùng **hiệu số theo từng fold** (mục 4 của file đó): hai mô hình được đánh giá trên cùng một tập test, nên hiệu số phản ánh chênh lệch thật giữa mô hình, không lẫn với biến động do cách chia dữ liệu.

## Tóm tắt

| Giả thuyết | Kết luận | Căn cứ chính |
| :-- | :-- | :-- |
| **H1** – Đầu kỳ (T ≤ 2), DBN ổn định hơn hoặc tương đương LSTM | **Ủng hộ một phần** | DBN ổn định hơn giữa các fold và F1 tương đương, nhưng AUC-ROC thấp hơn LSTM ở 5/5 fold |
| **H2** – Từ T ≥ 4, LSTM có F1 và AUC-ROC cao hơn DBN | **Được ủng hộ** (so với DBN) | AUC-ROC cao hơn ở 5/5 fold tại mọi mốc T ≥ 4; F1 cao hơn ở 4/5 fold (T = 4) và 5/5 fold (T ≥ 8) |
| ↳ Lý do "nhờ học được chuỗi hành vi" | **Chưa được ủng hộ** | Random Forest chỉ dùng đặc trưng tích lũy đến T đạt AUC-ROC ngang LSTM ở mọi mốc tuần |
| **H3** – DBN cung cấp giải thích các yếu tố ảnh hưởng | **Được ủng hộ** (định tính) | DBN trả về ΔP cho từng yếu tố của từng sinh viên; LSTM không có sẵn khả năng này |

---

## H1. Đầu kỳ (T = 2): DBN so với LSTM

| Chỉ số | DBN | LSTM | DBN − LSTM (theo fold) | Số fold DBN > LSTM |
| :-- | :-: | :-: | :-: | :-: |
| AUC-ROC | 0.6282 ± 0.0043 | 0.6601 ± 0.0064 | −0.0319 ± 0.0087 | 0/5 |
| AUC-PR | 0.5210 ± 0.0056 | 0.5629 ± 0.0097 | – | – |
| F1 | 0.5636 ± **0.0091** | 0.5439 ± **0.0243** | +0.0198 ± 0.0300 | 3/5 |
| Recall | 0.6754 ± **0.0262** | 0.5616 ± **0.0602** | +0.1138 ± 0.0787 | 5/5 |
| Precision | 0.4838 | 0.5304 | – | – |

**Phần được ủng hộ:**
- **Ổn định hơn:** độ lệch chuẩn giữa các fold của DBN nhỏ hơn LSTM ở mọi chỉ số; với F1 và Recall, độ lệch của DBN chỉ bằng khoảng 1/3–1/2 của LSTM.
- **Tương đương về F1:** chênh lệch +0.020 nhỏ hơn độ lệch chuẩn của chính nó (0.030), và DBN chỉ cao hơn ở 3/5 fold.

**Phần không được ủng hộ:**
- **Khả năng xếp hạng nguy cơ (AUC-ROC, AUC-PR):** DBN thấp hơn LSTM ở cả 5 fold. Hai chỉ số này không phụ thuộc ngưỡng, nên là thước đo công bằng nhất về chất lượng mô hình.
- **Recall cao hơn không có nghĩa là mô hình tốt hơn:** DBN đạt Recall cao hơn vì chọn ngưỡng nghiêng về phía cảnh báo (Precision 0.484 so với 0.530). Đây là đánh đổi ở bước chọn ngưỡng, không phải ưu thế phân biệt của mô hình.
- **DBN không phải mô hình tốt nhất ở T = 2:** cả hai mô hình đối chứng đều có AUC-ROC cao hơn (LR 0.6678, RF 0.6667).

**Nguyên nhân có thể (cần kiểm chứng thêm):**
1. **Đến hết tuần 2 gần như chưa có dữ liệu học tập.** Chỉ **4.7%** lượt đăng ký có bài đánh giá đến hạn và 14.2% có bất kỳ hoạt động nộp bài nào. Ở T = 2, mọi mô hình về thực chất chỉ dự đoán từ đặc trưng nền (nhân khẩu học, học vấn, môn học).
2. **Rời rạc hóa làm DBN mất thông tin nền.** DBN dùng các nhóm thô (IMD 4 mức, tín chỉ 3 mức, số lần học lại 3 mức), trong khi LR/RF dùng one-hot đầy đủ và giá trị số liên tục. Khi chỉ còn đặc trưng nền, mất mát này thể hiện rõ nhất.
3. **Pipeline chưa dùng dữ liệu click LMS (`studentVle`).** H1 phát biểu cho tình huống "dữ liệu LMS còn ít", nhưng hệ thống hiện chưa đưa click vào mô hình nào. Vì vậy H1 mới chỉ được kiểm chứng trên dữ liệu bài đánh giá, chưa đúng như phát biểu ban đầu.

---

## H2. Từ giữa kỳ (T ≥ 4): LSTM so với DBN

| Tuần | ΔAUC-ROC (LSTM − DBN) | Số fold LSTM > DBN | ΔF1 (LSTM − DBN) | Số fold LSTM > DBN |
| :-: | :-: | :-: | :-: | :-: |
| 4 | +0.0519 | 5/5 | +0.0331 | 4/5 |
| 8 | +0.0427 | 5/5 | +0.0254 | 5/5 |
| 12 | +0.0356 | 5/5 | +0.0245 | 5/5 |
| 16 | +0.0216 | 5/5 | +0.0077 | 5/5 |
| 20 | +0.0187 | 5/5 | +0.0097 | 5/5 |

**H2 được ủng hộ:** LSTM vượt DBN nhất quán ở mọi mốc tuần, mọi fold. Khoảng cách lớn nhất ở T = 4 rồi thu hẹp dần về cuối kỳ, khi dữ liệu tích lũy đủ nhiều để cả hai mô hình cùng phân biệt tốt.

**Tuy nhiên, lý do mà H2 nêu ("nhờ học được chuỗi hành vi phức tạp") chưa được ủng hộ:**

| Tuần | ΔAUC-ROC (LSTM − RF) | Số fold LSTM > RF | ΔAUC-ROC (LSTM − LR) | Số fold LSTM > LR |
| :-: | :-: | :-: | :-: | :-: |
| 4 | −0.0048 | 1/5 | +0.0092 | 5/5 |
| 8 | −0.0022 | 2/5 | +0.0048 | 5/5 |
| 12 | −0.0011 | 2/5 | +0.0077 | 5/5 |
| 16 | −0.0004 | 3/5 | +0.0097 | 5/5 |
| 20 | −0.0018 | 1/5 | +0.0045 | 4/5 |

- Random Forest chỉ nhìn bảng đặc trưng tích lũy đến tuần T, không có thông tin thứ tự thời gian, nhưng đạt AUC-ROC ngang hoặc nhỉnh hơn LSTM. Ở T = 16 và T = 20, F1 của LSTM thấp hơn RF ở cả 5/5 fold.
- LSTM chỉ vượt Logistic Regression một khoảng nhỏ (+0.005 đến +0.010 AUC-ROC), tức là lợi ích chủ yếu đến từ việc mô hình hóa phi tuyến, mà RF cũng làm được.

**Nguyên nhân có thể:**
1. **Chuỗi đầu vào của LSTM đã chứa sẵn đặc trưng tích lũy.** 10/15 kênh trong tensor tuần là đại lượng tích lũy đến tuần t (số bài đã nộp, số bài bỏ, điểm trung bình trọng số…), giống thông tin RF nhận được. Phần "chuỗi" thực sự bổ sung chỉ là thời điểm xảy ra sự kiện.
2. **Chuỗi tuần rất thưa.** Đến tuần 20, trung bình mỗi lượt đăng ký chỉ có 4.4 bài đánh giá định kỳ đến hạn; phần lớn các tuần không có sự kiện nào. Dữ liệu click LMS hằng tuần mới là loại chuỗi dày mà LSTM có lợi thế, nhưng hiện chưa được sử dụng.

**Ý nghĩa với thiết kế DSS:** việc chọn LSTM làm mô hình dự báo chính nên được lập luận bằng **lợi thế vận hành**, không phải độ chính xác:
- Một mô hình duy nhất cho xác suất ở mọi tuần 1..20, nên vẽ được đường xu hướng rủi ro cho từng sinh viên.
- Baseline cần huấn luyện riêng một mô hình cho từng mốc tuần.

---

## H3. Khả năng giải thích của DBN

DBN trả về độ đóng góp ΔP của từng yếu tố cho từng sinh viên tại tuần T. Đây chính là thông tin hiển thị ở tab "Chẩn đoán & Giải thích" của dashboard. Ví dụ sinh viên `74372` (môn AAA, 2013J) tại T = 8: P(At-risk) = 85.9%, yếu tố tăng rủi ro mạnh nhất là điểm tuần 5–8 dưới 40 (`perf_S3 = F_lt40`, +36.4%). Kết quả thực tế của sinh viên này là Fail.

**Giới hạn cần nêu trong báo cáo:**
- **ΔP là giải thích theo mô hình, không phải tác động nhân quả theo nghĩa can thiệp.** ΔP đo mức thay đổi xác suất khi bỏ một bằng chứng khỏi truy vấn. Cấu trúc mạng được học từ dữ liệu quan sát, nên không khẳng định được rằng thay đổi yếu tố đó sẽ thay đổi kết quả.
- **H3 mới được kiểm chứng định tính.** Chưa có đánh giá với người dùng (cố vấn học tập) về việc giải thích có hữu ích cho quyết định can thiệp hay không.

---

## Hệ quả với mô hình Hybrid (RQ3)

- **Trọng số đầu kỳ chưa có căn cứ.** Hybrid trong dashboard đặt trọng số DBN 60% khi T ≤ 2, dựa trên giả định của H1. Kết quả trên cho thấy ở T = 2, DBN có AUC-ROC thấp nhất trong 4 mô hình, nên trọng số này chưa có căn cứ thực nghiệm.
- **Hybrid chưa được đánh giá.** Mô hình Hybrid chưa nằm trong benchmark, nên RQ3 về hiệu năng dự báo vẫn còn mở.

## Giới hạn của phân tích

- **5 fold chưa phải kiểm định thống kê chính thức.** Kết quả "5/5 fold" là bằng chứng nhất quán mạnh nhưng không thay được kiểm định đó; các chênh lệch nhỏ (dưới 0.005 AUC-ROC) nên xem là tương đương.
- **Kết quả gắn với một cấu hình cụ thể.** Mỗi mô hình dùng một bộ siêu tham số cố định, và đặc trưng chỉ lấy từ bài đánh giá; kết luận có thể thay đổi khi bổ sung dữ liệu click LMS.
- **Mẫu thiếu nhóm sinh viên rủi ro cao nhất.** Danh sách lượt đăng ký được lấy từ các lượt có ít nhất một bài nộp, nên sinh viên rút môn trước khi nộp bài nào không nằm trong mẫu đánh giá.
