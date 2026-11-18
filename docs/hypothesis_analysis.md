# Phân tích kiểm chứng giả thuyết H1, H2, H3 và RQ3

Số liệu lấy từ kiểm định chéo 5-fold (`StratifiedGroupKFold` theo `id_student`) trong [experimental_results.md](experimental_results.md), cùng kiểm tra theo thời gian trong [temporal_holdout.md](temporal_holdout.md).

**Mẫu đánh giá:**
- **29.915 lượt đăng ký** (26.740 sinh viên, tỷ lệ At-risk 48.57%): toàn bộ lượt đăng ký trong `studentInfo.csv`, trừ sinh viên hủy môn trước ngày bắt đầu khóa học.
- Bao gồm cả 4.101 lượt chưa nộp bài nào; trong nhóm này 99.9% là Withdrawn hoặc Fail.

**Đặc trưng:** cả 5 mô hình dùng chung hai nguồn dữ liệu theo tuần:
- Bài đánh giá: điểm, nộp trễ, bỏ bài quá hạn.
- Click LMS: số click và số ngày truy cập mỗi tuần, gộp từ `studentVle.csv`.

**Quy trình đánh giá thống nhất:**
- Trong mỗi fold, 15% sinh viên của tập Train được tách làm **tập Validation chung**. Tập này dùng để dừng sớm LSTM, chọn ngưỡng phân loại tối ưu F1 cho **mọi mô hình** (lưới 0.20–0.80, bước 0.01) và chọn trọng số Hybrid.
- Vì mọi mô hình chọn ngưỡng theo cùng một cách, các so sánh F1, Recall, Accuracy giữa các mô hình là công bằng.
- Cấu trúc DBN học bằng Hill Climbing phụ thuộc thứ tự duyệt biến, tức phụ thuộc `PYTHONHASHSEED`. Script benchmark tự chạy lại với `PYTHONHASHSEED=0`, nên kết quả lặp lại được.

Các so sánh giữa hai mô hình dùng **hiệu số theo từng fold** (mục 4 của file kết quả). Hai mô hình được đánh giá trên cùng một tập test, nên hiệu số phản ánh chênh lệch thật giữa mô hình, không lẫn với biến động do cách chia dữ liệu.

## Tóm tắt

| Giả thuyết / Câu hỏi | Kết luận | Căn cứ chính |
| :-- | :-- | :-- |
| **H1** – Đầu kỳ (T ≤ 2), DBN ổn định hơn hoặc tương đương LSTM | **Không được ủng hộ** | AUC-ROC và F1 của DBN thấp hơn LSTM ở 5/5 fold; độ lệch chuẩn giữa các fold của DBN lớn hơn ở 4/6 chỉ số |
| **H2** – Từ T ≥ 4, LSTM có F1 và AUC-ROC cao hơn DBN | **Được ủng hộ** | AUC-ROC và F1 của LSTM cao hơn DBN ở 5/5 fold tại mọi mốc T ≥ 4 |
| ↳ Lý do "nhờ học được chuỗi hành vi" | **Ủng hộ rất yếu** | LSTM nhỉnh hơn Random Forest (chỉ dùng đặc trưng tích lũy) ở 3–5/5 fold, nhưng chênh lệch AUC-ROC dưới 0.004 và F1 gần như bằng nhau |
| **H3** – DBN cung cấp giải thích các yếu tố ảnh hưởng | **Được ủng hộ** (định tính) | DBN trả về ΔP cho từng yếu tố của từng sinh viên, kể cả mức truy cập LMS; LSTM không có sẵn khả năng này |
| **RQ3** – Hybrid DBN + LSTM có tốt hơn từng mô hình riêng | **Ngang LSTM** | Trọng số DBN chọn trên Validation chỉ 0–0.3; AUC-ROC của Hybrid cao hơn LSTM tối đa 0.001 |

---

## Tác động của đặc trưng click LMS

So sánh với lần chạy trước khi bổ sung click: cùng mẫu 29.915 lượt, cùng cách chia fold. Mỗi ô là hiệu số AUC-ROC trung bình (có click − không click), trong ngoặc là số fold được cải thiện. Số liệu đo ở lần chạy bổ sung click, trước khi thống nhất tập Validation; AUC-ROC không phụ thuộc ngưỡng nên kết luận không đổi.

| Tuần | DBN | LSTM | RF | LR |
| :-: | :-: | :-: | :-: | :-: |
| 2 | +0.0873 (5/5) | +0.0815 (5/5) | +0.0752 (5/5) | +0.0790 (5/5) |
| 4 | +0.0243 (5/5) | +0.0322 (5/5) | +0.0284 (5/5) | +0.0320 (5/5) |
| 8 | +0.0076 (5/5) | +0.0152 (5/5) | +0.0115 (5/5) | +0.0141 (5/5) |
| 12 | +0.0061 (5/5) | +0.0083 (5/5) | +0.0058 (5/5) | +0.0097 (5/5) |
| 16 | +0.0108 (5/5) | +0.0097 (5/5) | +0.0073 (5/5) | +0.0095 (5/5) |
| 20 | +0.0083 (5/5) | +0.0067 (5/5) | +0.0047 (5/5) | +0.0056 (5/5) |

- **Click cải thiện mọi mô hình ở mọi mốc tuần, trên mọi fold.**
- **Mức cải thiện lớn nhất ở T = 2.** Đến hết tuần 2, 92.1% lượt đăng ký đã có click LMS, trong khi chỉ 4.5% có bài đánh giá đến hạn. Click là nguồn tín hiệu học tập gần như duy nhất ở đầu kỳ.
- **Mức cải thiện giảm dần về cuối kỳ**, khi điểm số và việc nộp bài đã mang phần lớn thông tin.

---

## H1. Đầu kỳ (T = 2): DBN so với LSTM

| Chỉ số | DBN | LSTM | DBN − LSTM (theo fold) | Số fold DBN > LSTM |
| :-- | :-: | :-: | :-: | :-: |
| AUC-ROC | 0.7083 ± 0.0131 | 0.7560 ± 0.0086 | −0.0477 ± 0.0137 | 0/5 |
| AUC-PR | 0.6956 ± 0.0204 | 0.7518 ± 0.0123 | – | – |
| F1 | 0.6792 ± 0.0029 | 0.6979 ± 0.0039 | −0.0187 ± 0.0035 | 0/5 |
| Recall | 0.8538 ± 0.0523 | 0.8290 ± 0.0487 | +0.0248 ± 0.0400 | 4/5 |
| Precision | 0.5657 ± 0.0224 | 0.6045 ± 0.0247 | – | – |
| Accuracy | 0.6083 ± 0.0234 | 0.6513 ± 0.0230 | – | – |

**H1 không được ủng hộ:**
- **Không tương đương:** AUC-ROC (−0.048) và F1 (−0.019) của DBN đều thấp hơn LSTM ở cả 5 fold.
- **Không ổn định hơn:** độ lệch chuẩn giữa các fold của DBN lớn hơn LSTM ở 4/6 chỉ số (Accuracy, Recall, AUC-ROC, AUC-PR); chỉ F1 và Precision là DBN dao động ít hơn.
- **Recall cao hơn không có nghĩa là mô hình tốt hơn:** Recall 0.85 của DBN đi kèm Precision và Accuracy thấp nhất trong 5 mô hình. DBN đang cảnh báo rộng tay hơn, đây là đánh đổi ở bước chọn ngưỡng, không phải ưu thế phân biệt.
- **DBN thấp nhất trong 5 mô hình:** ở T = 2, bốn mô hình còn lại gần như ngang nhau về AUC-ROC (LR 0.7581, Hybrid 0.7569, LSTM 0.7560, RF 0.7532), còn DBN thấp hơn rõ (0.7083).

**Nguyên nhân có thể (liên quan đến thiết kế DBN, cần kiểm chứng thêm):**
1. **Rời rạc hóa làm mất thông tin.** Mỗi lát cắt, click chỉ được chia 4 mức (không truy cập / thấp / trung bình / cao), và đặc trưng nền cũng được gộp thành nhóm thô (IMD 4 mức, tín chỉ 3 mức). Các mô hình còn lại dùng giá trị liên tục: log số click, số ngày truy cập, one-hot đầy đủ.
2. **Đường truyền thông tin từ đầu kỳ tới nút đích bị suy giảm.** Ở T = 2, thông tin click tuần 1–2 phải đi qua chuỗi 5 nút click tương lai chưa quan sát (`click_S2` → … → `click_S6`) mới tới `at_risk`. Mỗi bước chuyển qua một bảng xác suất 4 trạng thái, nên tín hiệu bị "pha loãng". Nút `at_risk` đã dùng hết số cha tối đa (4) cho các cạnh bắt buộc, nên thuật toán học cấu trúc không thể thêm cạnh trực tiếp từ bằng chứng đầu kỳ.
3. **Hướng cải thiện:** tăng số mức rời rạc hóa cho click, nới giới hạn số cha của `at_risk`, hoặc dùng mạng chuyển tiếp 2-TBN.

---

## H2. Từ giữa kỳ (T ≥ 4): LSTM so với DBN

| Tuần | ΔAUC-ROC (LSTM − DBN) | Số fold LSTM > DBN | ΔF1 (LSTM − DBN) | Số fold LSTM > DBN |
| :-: | :-: | :-: | :-: | :-: |
| 4 | +0.0525 | 5/5 | +0.0322 | 5/5 |
| 8 | +0.0504 | 5/5 | +0.0444 | 5/5 |
| 12 | +0.0360 | 5/5 | +0.0308 | 5/5 |
| 16 | +0.0177 | 5/5 | +0.0151 | 5/5 |
| 20 | +0.0147 | 5/5 | +0.0121 | 5/5 |

**H2 được ủng hộ:** LSTM vượt DBN cả về AUC-ROC lẫn F1, nhất quán ở mọi mốc tuần, mọi fold. Khoảng cách lớn nhất ở giai đoạn T = 4–8 rồi thu hẹp dần về cuối kỳ, khi dữ liệu tích lũy đủ nhiều để cả hai mô hình cùng phân biệt tốt.

**Lý do mà H2 nêu ("nhờ học được chuỗi hành vi phức tạp") chỉ được ủng hộ rất yếu:**

| Tuần | ΔAUC-ROC (LSTM − RF) | Số fold LSTM > RF | ΔF1 (LSTM − RF) | ΔAUC-ROC (LSTM − LR) | Số fold LSTM > LR |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 4 | +0.0005 | 3/5 | −0.0012 | +0.0074 | 5/5 |
| 8 | +0.0033 | 4/5 | +0.0019 | +0.0050 | 5/5 |
| 12 | +0.0032 | 4/5 | +0.0012 | +0.0057 | 5/5 |
| 16 | +0.0030 | 5/5 | +0.0053 | +0.0078 | 5/5 |
| 20 | +0.0010 | 4/5 | +0.0001 | +0.0052 | 5/5 |

- **LSTM nhỉnh hơn RF nhưng chênh lệch rất nhỏ.** Random Forest chỉ nhìn bảng đặc trưng tích lũy đến tuần T, không có thông tin thứ tự thời gian. LSTM vượt RF ở 3–5/5 fold, nhưng chênh lệch AUC-ROC luôn dưới 0.004, không có ý nghĩa thực tế với một DSS.
- **F1 gần như bằng nhau.** Khi mọi mô hình chọn ngưỡng trên cùng tập Validation, chênh lệch F1 giữa LSTM và RF chỉ trong khoảng ±0.005. Khoảng cách F1 lớn hơn thấy ở phiên bản trước (khi RF dùng ngưỡng cố định 0.5) chủ yếu do cách chọn ngưỡng.
- **LSTM vượt LR ổn định hơn** (5/5 fold từ T ≥ 4), nhưng chênh lệch cũng dưới 0.008.

**Nguyên nhân khiến khoảng cách nhỏ:**
1. **Chuỗi đầu vào của LSTM đã chứa sẵn đặc trưng tích lũy.** 11/18 kênh trong tensor tuần là đại lượng tích lũy đến tuần t (số bài đã nộp, số bài bỏ, điểm trung bình trọng số, tổng click…), giống thông tin RF nhận được. Phần "chuỗi" thực sự bổ sung chỉ là thời điểm và nhịp độ của sự kiện.
2. **Chuỗi bài đánh giá rất thưa.** Đến tuần 20, trung bình mỗi lượt đăng ký chỉ có 4.4 bài đánh giá định kỳ đến hạn.

**Ý nghĩa với thiết kế DSS:** việc chọn LSTM làm mô hình dự báo chính được lập luận tốt nhất bằng **lợi thế vận hành và hiệu chỉnh xác suất**, không phải độ chính xác:
- Một mô hình duy nhất cho xác suất ở mọi tuần 1..20, nên vẽ được đường xu hướng rủi ro cho từng sinh viên.
- Baseline cần huấn luyện riêng một mô hình cho từng mốc tuần.
- Xác suất của LSTM sát tỷ lệ thực tế (ECE dưới 0.02 ở mọi mốc tuần, xem mục Hiệu chỉnh xác suất).
- Về độ chính xác, LSTM chỉ ngang hoặc nhỉnh hơn rất ít so với mô hình đối chứng.

---

## H3. Khả năng giải thích của DBN

DBN trả về độ đóng góp ΔP của từng yếu tố cho từng sinh viên tại tuần T, gồm cả mức truy cập LMS theo lát cắt (`click_S1..S6`). Đây chính là thông tin hiển thị ở tab "Chẩn đoán & Giải thích" của dashboard.

**Ví dụ:** sinh viên `1456619` (môn AAA, 2013J) tại T = 8 có P(At-risk) theo DBN = 90.9%. Kết quả thực tế của sinh viên này là Withdrawn.

| Yếu tố | Giá trị | ΔP |
| :-- | :-- | :-: |
| `click_S3` | `C0_None` (không truy cập LMS tuần 5–8) | +16.3% |
| `code_module` | `AAA` | −6.9% |
| `age_band` | `35-55` | +0.6% |
| `highest_education` | `A Level or Equivalent` | +0.4% |
| `imd_group` | `High_60_100` | +0.2% |

Ở ví dụ này, giải thích dễ hiểu: việc ngừng truy cập LMS trong cả 4 tuần gần nhất là yếu tố đẩy rủi ro lên mạnh nhất, trong khi môn AAA (tỷ lệ At-risk thấp nhất trong 7 môn, 29%) kéo rủi ro xuống.

**Giới hạn cần nêu trong báo cáo:**
- **ΔP là giải thích theo mô hình, không phải tác động nhân quả theo nghĩa can thiệp.** ΔP đo mức thay đổi xác suất khi bỏ một bằng chứng khỏi truy vấn. Cấu trúc mạng được học từ dữ liệu quan sát, nên không khẳng định được rằng thay đổi yếu tố đó sẽ thay đổi kết quả.
- **ΔP phụ thuộc vào các bằng chứng còn lại.** ΔP nhỏ không có nghĩa là yếu tố không quan trọng, vì các yếu tố mang thông tin trùng nhau sẽ bù cho nhau. Ở một số sinh viên, dấu của ΔP có thể ngược trực giác (ví dụ mức click thấp lại giảm nhẹ rủi ro khi sinh viên đã bỏ bài trong cùng giai đoạn).
- **H3 mới được kiểm chứng định tính.** Chưa có đánh giá với người dùng (cố vấn học tập) về việc giải thích có hữu ích cho quyết định can thiệp hay không.

---

## RQ3. Mô hình Hybrid (DBN + LSTM)

P(Hybrid) = a · P(DBN) + (1 − a) · P(LSTM), với a ∈ {0, 0.1, …, 1} chọn theo AUC-ROC trên tập Validation, riêng cho từng mốc tuần.

| Tuần | Trọng số DBN (5 fold) | ΔAUC-ROC (Hybrid − LSTM) | Số fold Hybrid > LSTM | ΔF1 (Hybrid − LSTM) |
| :-: | :-: | :-: | :-: | :-: |
| 2 | 0.1–0.2 | +0.0010 | 5/5 | +0.0006 |
| 4 | 0–0.1 | +0.0001 | 1/5 | −0.0007 |
| 8 | 0–0.1 | +0.0005 | 3/5 | +0.0021 |
| 12 | 0–0.1 | +0.0005 | 4/5 | +0.0012 |
| 16 | 0.1 | +0.0005 | 5/5 | −0.0005 |
| 20 | 0.1–0.3 | +0.0005 | 5/5 | +0.0014 |

- **Hybrid gần như trùng LSTM.** Trọng số DBN được chọn chỉ 0–0.3, nên Hybrid thêm được tối đa 0.001 AUC-ROC. Chênh lệch nhất quán (5/5 fold ở T = 2, 16, 20) nhưng không có ý nghĩa thực tế.
- **Trọng số đầu kỳ ngược với giả định ban đầu.** Thiết kế ban đầu đặt trọng số DBN 60% khi T ≤ 2 dựa trên H1. Dữ liệu cho thấy trọng số tối ưu chỉ 0.1–0.2, phù hợp với việc H1 không được ủng hộ. Hệ thống hiện đọc trọng số đã chọn trên Validation (`models/hybrid_weights.json`) thay cho trọng số cố định.
- **Giá trị chính của DBN trong Hybrid là phần giải thích (H3)**, không phải độ chính xác dự báo.

---

## Hiệu chỉnh xác suất (Calibration)

DSS hiển thị xác suất At-risk và chia mức rủi ro theo ngưỡng xác suất, nên xác suất cần sát tỷ lệ thực tế, không chỉ xếp hạng đúng.

| Tuần | ECE DBN | ECE LSTM | ECE Hybrid | ECE RF | ECE LR |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 2 | 0.0532 | 0.0174 | 0.0225 | 0.0361 | 0.0186 |
| 4 | 0.0624 | 0.0161 | 0.0158 | 0.0276 | 0.0177 |
| 8 | 0.0477 | 0.0176 | 0.0182 | 0.0233 | 0.0133 |
| 12 | 0.0380 | 0.0181 | 0.0177 | 0.0214 | 0.0172 |
| 16 | 0.0175 | 0.0184 | 0.0177 | 0.0180 | 0.0182 |
| 20 | 0.0141 | 0.0172 | 0.0162 | 0.0174 | 0.0170 |

- **LSTM và Hybrid hiệu chỉnh tốt:** ECE dưới 0.023 ở mọi mốc tuần; Brier score thấp nhất (hoặc gần thấp nhất) trong 5 mô hình ở mọi mốc tuần.
- **DBN lệch nhiều ở đầu và giữa kỳ** (ECE 0.038–0.062 tại T = 2–12), sau đó tốt dần và hiệu chỉnh tốt nhất ở T = 16–20. Điều này phù hợp với nhận định ở H1: khi bằng chứng đầu kỳ ít, phân phối hậu nghiệm của DBN bị kéo về xác suất tiên nghiệm.
- **Ngưỡng tối ưu F1 thấp hơn 0.5 ở đầu kỳ** (0.34–0.42 tại T = 2), tăng dần về 0.43–0.53 ở T = 20. Ở đầu kỳ, mô hình chưa chắc chắn nên hạ ngưỡng để bắt được nhiều sinh viên At-risk hơn.

---

## Tổng quát theo thời gian (học 2013 → đánh giá 2014)

Kiểm định chéo 5-fold trộn mọi kỳ học, nên chưa trả lời được câu hỏi thực tế: mô hình học từ các khóa cũ có dùng được cho khóa mới không. Thí nghiệm bổ sung học trên các kỳ 2013B, 2013J (12.555 lượt) và đánh giá trên 2014B, 2014J (13.389 lượt), loại môn CCC vì chỉ có ở 2014.

| Tuần | DBN | LSTM | Hybrid | RF | LR |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 2 | 0.7068 (−0.002) | 0.7511 (−0.005) | 0.7527 (−0.004) | 0.7521 (−0.001) | **0.7543** (−0.004) |
| 4 | 0.7537 (−0.007) | 0.7786 (−0.034) | **0.7806** (−0.032) | 0.7795 (−0.033) | 0.7709 (−0.034) |
| 8 | 0.8424 (+0.025) | 0.8419 (−0.026) | 0.8468 (−0.022) | 0.8418 (−0.023) | **0.8594** (−0.004) |
| 12 | 0.8593 (+0.002) | **0.8852** (−0.009) | 0.8849 (−0.009) | 0.8696 (−0.021) | 0.8765 (−0.012) |
| 16 | 0.9037 (−0.003) | 0.9167 (−0.008) | **0.9185** (−0.007) | 0.9075 (−0.014) | 0.9123 (−0.005) |
| 20 | 0.9211 (−0.009) | 0.9413 (−0.004) | **0.9418** (−0.004) | 0.9377 (−0.007) | 0.9385 (−0.002) |

*Mỗi ô: AUC-ROC trên 2014, trong ngoặc là chênh lệch so với 5-fold.*

- **Mức giảm nhẹ:** LSTM và Hybrid giảm dưới 0.01 AUC-ROC ở T = 2 và T ≥ 12; RF giảm nhiều hơn ở cuối kỳ (0.014–0.021 tại T = 12–16). Mức giảm lớn nhất ở T = 4–8 (khoảng 0.02–0.03 với LSTM, Hybrid, RF; LR giảm 0.034 tại T = 4). Nguyên nhân cụ thể (ví dụ lịch bài đánh giá thay đổi giữa các năm) chưa được kiểm chứng.
- **DBN gần như không giảm**, thậm chí tăng ở T = 8. Mô hình rời rạc hóa thô có thể ít bị ảnh hưởng hơn khi phân phối dữ liệu thay đổi, nhưng vẫn thấp hơn LSTM/Hybrid ở hầu hết các mốc tuần.
- **Thứ hạng giữa các mô hình gần như giữ nguyên:** LSTM/Hybrid tốt nhất từ T ≥ 12, các mô hình còn lại ngang nhau ở đầu kỳ.
- **Giới hạn:** tập học chỉ bằng khoảng một nửa so với 5-fold, nên mức giảm là kết hợp của thay đổi theo thời gian và ít dữ liệu học hơn. Đây là một lần chia duy nhất, không có độ lệch chuẩn; 716 sinh viên học lại có mặt ở cả hai năm.

---

## Giới hạn của phân tích

- **5 fold chưa phải kiểm định thống kê chính thức.** Kết quả "5/5 fold" là bằng chứng nhất quán mạnh nhưng không thay được kiểm định đó; các chênh lệch nhỏ (dưới 0.005 AUC-ROC) nên xem là tương đương.
- **Phân vị chia mức click của DBN được tính trên toàn bộ mẫu**, kể cả tập test của từng fold. Cách tính này không dùng nhãn, nên ảnh hưởng dự kiến nhỏ, nhưng về nguyên tắc nên tính lại trên tập train của từng fold.
- **Kết quả gắn với một cấu hình cụ thể.** Mỗi mô hình dùng một bộ siêu tham số cố định; kết luận về DBN có thể thay đổi khi thiết kế lại cách rời rạc hóa và cấu trúc mạng.
- **Chống rò rỉ thông tin tương lai đã được kiểm thử tự động** (`tests/test_leakage.py`): thay đổi mọi dữ liệu sau tuần T không làm thay đổi đặc trưng tại tuần T của cả ba nhóm mô hình. Kiểm thử chỉ bao phủ bước tạo đặc trưng và đầu ra LSTM, chưa bao phủ toàn bộ quá trình huấn luyện.
- **Không so trực tiếp với số liệu của các phiên bản trước.** Phiên bản đầu tiên chỉ đánh giá 25.814 lượt có ít nhất một bài nộp, không có click LMS, và mô hình đối chứng dùng ngưỡng cố định 0.5, nên các chỉ số tuyệt đối không so sánh trực tiếp được.
