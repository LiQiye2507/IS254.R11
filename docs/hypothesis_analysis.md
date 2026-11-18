# Phân tích kiểm chứng giả thuyết H1, H2, H3

Số liệu lấy từ kiểm định chéo 5-fold (`StratifiedGroupKFold` theo `id_student`) trong [experimental_results.md](experimental_results.md).

**Mẫu đánh giá:**
- **29.915 lượt đăng ký** (26.740 sinh viên, tỷ lệ At-risk 48.57%): toàn bộ lượt đăng ký trong `studentInfo.csv`, trừ sinh viên hủy môn trước ngày bắt đầu khóa học.
- Bao gồm cả 4.101 lượt chưa nộp bài nào; trong nhóm này 99.9% là Withdrawn hoặc Fail.

**Đặc trưng:** cả 4 mô hình dùng chung hai nguồn dữ liệu theo tuần:
- Bài đánh giá: điểm, nộp trễ, bỏ bài quá hạn.
- Click LMS: số click và số ngày truy cập mỗi tuần, gộp từ `studentVle.csv`.

Các so sánh giữa hai mô hình dùng **hiệu số theo từng fold** (mục 4 của file kết quả). Hai mô hình được đánh giá trên cùng một tập test, nên hiệu số phản ánh chênh lệch thật giữa mô hình, không lẫn với biến động do cách chia dữ liệu.

> **Lưu ý về ngưỡng:** DBN và LSTM chọn ngưỡng phân loại tối ưu F1 trên tập validation, còn mô hình đối chứng (RF, LR) dùng ngưỡng cố định 0.5. Vì vậy Accuracy, Precision, Recall và F1 phụ thuộc cả vào cách chọn ngưỡng. **AUC-ROC và AUC-PR không phụ thuộc ngưỡng**, nên là thước đo chính khi so sánh chất lượng mô hình.

## Tóm tắt

| Giả thuyết | Kết luận | Căn cứ chính |
| :-- | :-- | :-- |
| **H1** – Đầu kỳ (T ≤ 2), DBN ổn định hơn hoặc tương đương LSTM | **Không được ủng hộ** | AUC-ROC và F1 của DBN thấp hơn LSTM ở 5/5 fold; độ lệch chuẩn giữa các fold của DBN lớn hơn ở 5/6 chỉ số |
| **H2** – Từ T ≥ 4, LSTM có F1 và AUC-ROC cao hơn DBN | **Được ủng hộ** | AUC-ROC và F1 của LSTM cao hơn DBN ở 5/5 fold tại mọi mốc T ≥ 4 |
| ↳ Lý do "nhờ học được chuỗi hành vi" | **Ủng hộ rất yếu** | LSTM nhỉnh hơn Random Forest (chỉ dùng đặc trưng tích lũy) ở 4–5/5 fold từ T ≥ 8, nhưng chênh lệch AUC-ROC dưới 0.003 |
| **H3** – DBN cung cấp giải thích các yếu tố ảnh hưởng | **Được ủng hộ** (định tính) | DBN trả về ΔP cho từng yếu tố của từng sinh viên, kể cả mức truy cập LMS; LSTM không có sẵn khả năng này |

---

## Tác động của đặc trưng click LMS

So sánh với lần chạy trước khi bổ sung click: cùng mẫu 29.915 lượt, cùng cách chia fold. Mỗi ô là hiệu số AUC-ROC trung bình (có click − không click), trong ngoặc là số fold được cải thiện.

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
| AUC-ROC | 0.7144 ± 0.0068 | 0.7560 ± 0.0086 | −0.0416 ± 0.0032 | 0/5 |
| AUC-PR | 0.7025 ± 0.0130 | 0.7518 ± 0.0123 | – | – |
| F1 | 0.6817 ± 0.0042 | 0.6919 ± 0.0035 | −0.0102 ± 0.0057 | 0/5 |
| Recall | 0.8440 ± 0.0239 | 0.7384 ± 0.0199 | +0.1056 ± 0.0333 | 5/5 |
| Precision | 0.5723 ± 0.0163 | 0.6514 ± 0.0125 | – | – |
| Accuracy | 0.6169 ± 0.0170 | 0.6807 ± 0.0069 | – | – |

**H1 không được ủng hộ:**
- **Không tương đương:** AUC-ROC (−0.042) và F1 (−0.010) của DBN đều thấp hơn LSTM ở cả 5 fold.
- **Không ổn định hơn:** độ lệch chuẩn giữa các fold của DBN lớn hơn LSTM ở 5/6 chỉ số (Accuracy, Precision, Recall, F1, AUC-PR); chỉ AUC-ROC là DBN dao động ít hơn (0.0068 so với 0.0086).
- **Recall cao hơn không có nghĩa là mô hình tốt hơn:** Recall 0.84 của DBN đi kèm Precision thấp nhất (0.57) và Accuracy thấp nhất (0.62) trong 4 mô hình. DBN đang cảnh báo rộng tay hơn, đây là đánh đổi ở bước chọn ngưỡng, không phải ưu thế phân biệt của mô hình.
- **DBN thấp nhất trong 4 mô hình:** ở T = 2, ba mô hình còn lại gần như ngang nhau về AUC-ROC (LR 0.7583, LSTM 0.7560, RF 0.7538), còn DBN thấp hơn rõ (0.7144).

> **Ghi chú:** trước khi bổ sung click, kết quả ở T = 2 là "ủng hộ một phần" (F1 tương đương, dao động F1/Recall nhỏ hơn). Khi có đủ dữ liệu LMS, đúng như phát biểu của H1, ưu thế đó không còn.

**Nguyên nhân có thể (liên quan đến thiết kế DBN, cần kiểm chứng thêm):**
1. **Rời rạc hóa làm mất thông tin.** Mỗi lát cắt, click chỉ được chia 4 mức (không truy cập / thấp / trung bình / cao), và đặc trưng nền cũng được gộp thành nhóm thô (IMD 4 mức, tín chỉ 3 mức). Các mô hình còn lại dùng giá trị liên tục: log số click, số ngày truy cập, one-hot đầy đủ.
2. **Đường truyền thông tin từ đầu kỳ tới nút đích bị suy giảm.** Ở T = 2, thông tin click tuần 1–2 phải đi qua chuỗi 5 nút click tương lai chưa quan sát (`click_S2` → … → `click_S6`) mới tới `at_risk`. Mỗi bước chuyển qua một bảng xác suất 4 trạng thái, nên tín hiệu bị "pha loãng". Nút `at_risk` đã dùng hết số cha tối đa (4) cho các cạnh bắt buộc, nên thuật toán học cấu trúc không thể thêm cạnh trực tiếp từ bằng chứng đầu kỳ.
3. **Hướng cải thiện:** tăng số mức rời rạc hóa cho click, nới giới hạn số cha của `at_risk`, hoặc dùng mạng chuyển tiếp 2-TBN (đã ghi trong mục Hướng phát triển của kế hoạch).

---

## H2. Từ giữa kỳ (T ≥ 4): LSTM so với DBN

| Tuần | ΔAUC-ROC (LSTM − DBN) | Số fold LSTM > DBN | ΔF1 (LSTM − DBN) | Số fold LSTM > DBN |
| :-: | :-: | :-: | :-: | :-: |
| 4 | +0.0495 | 5/5 | +0.0304 | 5/5 |
| 8 | +0.0431 | 5/5 | +0.0358 | 5/5 |
| 12 | +0.0353 | 5/5 | +0.0340 | 5/5 |
| 16 | +0.0170 | 5/5 | +0.0126 | 5/5 |
| 20 | +0.0138 | 5/5 | +0.0088 | 5/5 |

**H2 được ủng hộ:** LSTM vượt DBN cả về AUC-ROC lẫn F1, nhất quán ở mọi mốc tuần, mọi fold. Khoảng cách lớn nhất ở giai đoạn T = 4–12 rồi thu hẹp dần về cuối kỳ, khi dữ liệu tích lũy đủ nhiều để cả hai mô hình cùng phân biệt tốt.

**Lý do mà H2 nêu ("nhờ học được chuỗi hành vi phức tạp") chỉ được ủng hộ rất yếu:**

| Tuần | ΔAUC-ROC (LSTM − RF) | Số fold LSTM > RF | ΔAUC-ROC (LSTM − LR) | Số fold LSTM > LR |
| :-: | :-: | :-: | :-: | :-: |
| 4 | −0.0005 | 2/5 | +0.0071 | 5/5 |
| 8 | +0.0023 | 4/5 | +0.0048 | 5/5 |
| 12 | +0.0029 | 4/5 | +0.0056 | 5/5 |
| 16 | +0.0026 | 5/5 | +0.0078 | 5/5 |
| 20 | +0.0008 | 4/5 | +0.0051 | 5/5 |

- **LSTM nhỉnh hơn RF nhưng chênh lệch rất nhỏ.** Random Forest chỉ nhìn bảng đặc trưng tích lũy đến tuần T, không có thông tin thứ tự thời gian. Từ T ≥ 8, LSTM vượt RF ở 4–5/5 fold, nhưng chênh lệch AUC-ROC luôn dưới 0.003, không có ý nghĩa thực tế với một DSS.
- **Click có lợi cho LSTM hơn một chút.** Từ T = 4 đến T = 12, LSTM được cải thiện từ click nhiều hơn RF (ví dụ T = 8: +0.015 so với +0.012). Điều này phù hợp với nhận định rằng chuỗi dày (click hằng tuần) là nơi LSTM có lợi thế.
- **F1 và Recall cao hơn của LSTM chưa so được.** LSTM có F1 cao hơn RF ở T ≤ 12 (5/5 fold), nhưng chủ yếu do khác biệt về ngưỡng, như lưu ý ở đầu tài liệu.

**Nguyên nhân khiến khoảng cách nhỏ:**
1. **Chuỗi đầu vào của LSTM đã chứa sẵn đặc trưng tích lũy.** 11/18 kênh trong tensor tuần là đại lượng tích lũy đến tuần t (số bài đã nộp, số bài bỏ, điểm trung bình trọng số, tổng click…), giống thông tin RF nhận được. Phần "chuỗi" thực sự bổ sung chỉ là thời điểm và nhịp độ của sự kiện.
2. **Chuỗi bài đánh giá rất thưa.** Đến tuần 20, trung bình mỗi lượt đăng ký chỉ có 4.4 bài đánh giá định kỳ đến hạn.

**Ý nghĩa với thiết kế DSS:** việc chọn LSTM làm mô hình dự báo chính được lập luận tốt nhất bằng **lợi thế vận hành**, không phải độ chính xác:
- Một mô hình duy nhất cho xác suất ở mọi tuần 1..20, nên vẽ được đường xu hướng rủi ro cho từng sinh viên.
- Baseline cần huấn luyện riêng một mô hình cho từng mốc tuần.
- Về độ chính xác, LSTM chỉ ngang hoặc nhỉnh hơn rất ít so với mô hình đối chứng.

---

## H3. Khả năng giải thích của DBN

DBN trả về độ đóng góp ΔP của từng yếu tố cho từng sinh viên tại tuần T, gồm cả mức truy cập LMS theo lát cắt (`click_S1..S6`). Đây chính là thông tin hiển thị ở tab "Chẩn đoán & Giải thích" của dashboard.

**Ví dụ:** sinh viên `58089` (môn BBB, 2013B) tại T = 8 có P(At-risk) = 74.6%. Kết quả thực tế của sinh viên này là Fail.

| Yếu tố | Giá trị | ΔP |
| :-- | :-- | :-: |
| `perf_S3` | `Missed_0` (bỏ bài tuần 5–8) | +18.2% |
| `code_module` | `BBB` | −5.8% |
| `prev_attempts_bin` | `2+` | +3.9% |
| `click_S3` | `C1_Low` | −2.4% |

**Giới hạn cần nêu trong báo cáo:**
- **ΔP là giải thích theo mô hình, không phải tác động nhân quả theo nghĩa can thiệp.** ΔP đo mức thay đổi xác suất khi bỏ một bằng chứng khỏi truy vấn. Cấu trúc mạng được học từ dữ liệu quan sát, nên không khẳng định được rằng thay đổi yếu tố đó sẽ thay đổi kết quả.
- **ΔP phụ thuộc vào các bằng chứng còn lại nên có thể khó diễn giải.** Ví dụ trên cho thấy mức click thấp ở tuần 5–8 lại *giảm nhẹ* rủi ro (−2.4%). Một cách giải thích có thể (chưa kiểm chứng): sinh viên này đã bỏ bài trong cùng giai đoạn. Khi đã biết điều đó, mức click thấp nhưng vẫn còn truy cập có thể là dấu hiệu tốt hơn so với trung bình các khả năng còn lại, trong đó có cả việc ngừng truy cập hoàn toàn. Tương tự, ΔP nhỏ không có nghĩa là yếu tố không quan trọng, vì các yếu tố mang thông tin trùng nhau sẽ bù cho nhau.
- **H3 mới được kiểm chứng định tính.** Chưa có đánh giá với người dùng (cố vấn học tập) về việc giải thích có hữu ích cho quyết định can thiệp hay không.

---

## Hệ quả với mô hình Hybrid (RQ3)

- **Trọng số đầu kỳ ngược với kết quả thực nghiệm.** Hybrid trong dashboard đặt trọng số DBN 60% khi T ≤ 2, dựa trên giả định của H1. Kết quả trên cho thấy ở T = 2, DBN có AUC-ROC thấp nhất trong 4 mô hình, và H1 không được ủng hộ, nên trọng số này cần được chọn lại trên tập validation.
- **Hybrid chưa được đánh giá.** Mô hình Hybrid chưa nằm trong benchmark, nên RQ3 về hiệu năng dự báo vẫn còn mở. Vai trò giải thích của DBN trong Hybrid (H3) vẫn đứng vững.

## Giới hạn của phân tích

- **5 fold chưa phải kiểm định thống kê chính thức.** Kết quả "5/5 fold" là bằng chứng nhất quán mạnh nhưng không thay được kiểm định đó; các chênh lệch nhỏ (dưới 0.005 AUC-ROC) nên xem là tương đương.
- **Cách chọn ngưỡng khác nhau giữa các mô hình.** Mô hình đối chứng dùng ngưỡng 0.5, DBN/LSTM dùng ngưỡng tối ưu trên validation, nên các so sánh F1/Recall giữa hai nhóm chưa hoàn toàn công bằng.
- **Phân vị chia mức click của DBN được tính trên toàn bộ mẫu**, kể cả tập test của từng fold. Cách tính này không dùng nhãn, nên ảnh hưởng dự kiến nhỏ, nhưng về nguyên tắc nên tính lại trên tập train của từng fold.
- **Kết quả gắn với một cấu hình cụ thể.** Mỗi mô hình dùng một bộ siêu tham số cố định; kết luận về DBN có thể thay đổi khi thiết kế lại cách rời rạc hóa và cấu trúc mạng.
- **Không so trực tiếp với số liệu của các phiên bản trước.** Phiên bản đầu tiên chỉ đánh giá 25.814 lượt có ít nhất một bài nộp và không có click LMS, nên các chỉ số tuyệt đối không so sánh trực tiếp được.
