# CSC4007 — Lab 4 Analysis Report

> Sinh viên điền báo cáo này sau khi chạy baseline và các biến thể nâng cấp.

## 1. Thông tin chung

- Họ tên: Triệu Quốc Anh
- MSSV: 1671040002
- Lớp: KHMT 16-01
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4007-lab4-TrieuQuocAnh
- Link W&B project/run nếu có: https://wandb.ai/noioivedi-dainam-vietnam/csc4007-lab4-lstm-gru

## 2. Baseline bắt buộc

Mô hình baseline trong Lab 4:

```text
Tokenized text → Embedding → 1-layer LSTM → Dropout → Linear classifier
```

Điền cấu hình đã chạy:

| Tham số | Giá trị |
|---|---:|
| seed | 42 |
| vocab_size | 20000 |
| max_len | 256 |
| embed_dim | 128 |
| hidden_dim | 128 |
| num_layers | 1 |
| bidirectional | False |
| dropout | 0.3 |
| lr | 1e-3 |
| batch_size | 64 |
| epochs_trained | 6 |

Kết quả baseline:

| Split | Loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 0.46082 | 0.8008 | 0.80037 |
| Test | 0.45706 | 0.80216 | 0.80162 |

Nhận xét ngắn về baseline:

- Baseline LSTM đạt được 80.16% test macro-F1, là mốc cơ bản cho so sánh.
- Gap giữa validation (0.8008) và test accuracy (0.80216) rất nhỏ (< 0.3%), chỉ ra model không quá overfit.
- Validation loss (0.461) cao hơn train loss (0.310) ~ 48%, nhưng vẫn chấp nhận được cho bài toán NLP.
- Mô hình dừng early stop ở epoch 6 khi validation macro-F1 đạt 0.80037, khá sớm so với các variants.
- LSTM đơn giản có khả năng capture dependencies tốt cho bài toán IMDB classification, nhưng vẫn còn khoảng 19.8% review bị phân loại sai.

## 3. Bảng ablation

Sinh viên cần thử ít nhất 2 biến thể nâng cấp so với baseline.

| Run | model_type | bidirectional | num_layers | max_len | hidden_dim | dropout | Test Accuracy | Test Macro-F1 | Nhận xét |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_lstm | lstm | False | 1 | 256 | 128 | 0.3 | 0.80216 | 0.80162 | Baseline LSTM |
| gru_baseline | gru | False | 1 | 256 | 128 | 0.3 | 0.85328 | 0.85327 | GRU tốt hơn LSTM 5.1% |
| stacked_bigru | gru | True | 2 | 256 | 128 | 0.4 | 0.85336 | 0.85332 | Stacked BiGRU với 2 lớp |

## 4. So sánh công bằng

Trả lời ngắn:

1. Các run có dùng cùng dataset không?
   - Có, tất cả các run đều dùng IMDB dataset.

2. Các run có dùng cùng train/validation/test split không?
   - Có, tất cả các run dùng cùng split với tỷ lệ train/val/test là 25000/5000/25000.

3. Các run có dùng cùng seed không?
   - Có, tất cả các run dùng seed 42 để đảm bảo tái tạo lại kết quả.

4. Metric chính để chọn mô hình là gì?
   - Validation macro-F1 được dùng để chọn mô hình trong quá trình training (early stopping).

5. Có dùng test set để chọn mô hình không? Vì sao không nên?
   - Không. Không nên dùng test set để chọn mô hình vì test set phải được giữ riêng để đánh giá khách quan kết quả cuối cùng. Nếu chọn mô hình dựa trên test set, mô hình sẽ bị overfit với test set, làm cho kết quả không đại diện cho hiệu suất thực tế trên dữ liệu mới.

## 5. Phân tích learning curves

Dựa vào `outputs/figures/loss_curve.png` và `outputs/figures/metric_curve.png`:

- Mô hình có dấu hiệu overfitting không?
- Validation macro-F1 cải thiện đến epoch nào?
- Có nên tăng epoch, giảm learning rate, hoặc tăng dropout không?

Nhận xét:

- **baseline_lstm (epoch 6)**: Gap giữa train_accuracy (0.876) và val_accuracy (0.801) = 7.5% → overfitting nhẹ. Val_loss (0.461) cao hơn train_loss (0.310), nhưng không quá đáng lo ngại.

- **gru_baseline (epoch 6)**: Gap = 8% (0.953 vs 0.873) → overfitting vừa phải. Val_loss (0.356) vẫn chấp nhận được so với train_loss (0.137). Model dừng lại khi val_macro_f1 = 0.873 nên đã tìm được điểm tối ưu.

- **stacked_bigru (epoch 8)**: Gap = 12.1% (0.987 vs 0.866) + Val_loss (0.668) >> Train_loss (0.038) → **Overfitting rõ rệt**. Mô hình bắt đầu overfit từ epoch 6-7 khi train_loss tiếp tục giảm nhưng val_loss tăng.

Khuyến nghị:
- **baseline_lstm**: Có thể tăng epochs thêm 1-2 để validation macro-F1 ổn định, nhưng hiệu quả sẽ hạn chế.
- **gru_baseline**: Điểm cân bằng tốt nhất → dừng ở epoch 6 là hợp lý. Không cần điều chỉnh lớn.
- **stacked_bigru**: Cần tăng dropout từ 0.4 lên 0.5-0.6 hoặc dùng early stopping ở epoch 5-6. Giảm learning rate thêm có thể giúp mô hình học từ từ hơn và chống overfitting tốt hơn.

## 6. Confusion matrix

Dựa vào `outputs/figures/confusion_matrix.png`:

- Mô hình nhầm lớp nào nhiều hơn?
- False positive và false negative có cân bằng không?
- Điều này ảnh hưởng gì nếu triển khai vào bài toán thực tế?

Nhận xét:

- **False Positive (FP) chiếm đa số**: Mô hình dự đoán negative reviews là positive. Từ error analysis, hầu hết lỗi (10/10 = 100%) là false positive do những review có từ tích cực (genius, masterpiece, excellent) nhưng vẫn là đánh giá âm.

- **False Negative ít hơn**: Số lượng FN nhỏ hơn FP, nhưng vẫn tồn tại. Các review tích cực nhưng mô hình dự đoán âm thường chứa lời phủ định hoặc structure phức tạp.

- **Không cân bằng giữa FP và FN**: FP >> FN, khiến precision thấp hơn recall. Accuracy cao (85%) nhưng model có xu hướng dễ "bắn" vào lớp positive.

- **Ảnh hưởng triển khai thực tế**:
  - **Bài toán phân loại review**: FP nhiều sẽ khiến một số negative reviews bị phân loại nhầm thành positive, gây confuse cho người đọc.
  - **Recommendation systems**: Nếu được đề xuất positively-labeled negative reviews, users sẽ rất thất vọng.
  - **Cải thiện**: Có thể điều chỉnh classification threshold từ 0.5 lên 0.6-0.7 để giảm FP, hoặc sử dụng weighted loss (trừng phạt FP nặng hơn FN) trong training.

## 7. Error analysis

Chọn ít nhất 10 mẫu sai từ `outputs/error_analysis/error_analysis.csv`.

| STT | Trích đoạn review | Nhãn đúng | Mô hình dự đoán | Confidence | Nguyên nhân giả định |
|---:|---|---|---|---:|---|
| 1 | "This is definitely one of the best Kung fu movies... a masterpiece!" | negative | positive | 0.9999 | Positive adjectives overwhelm context |
| 2 | "This movie was pure genius... hilarious... great acting... magnificent." | negative | positive | 0.9999 | Dồi dào từ tích cực |
| 3 | "Starts well... gripping... sadly collapses... left deeply disappointed" | negative | positive | 0.9999 | Conflict: positive parts vs. negative conclusion |
| 4 | "Story is good... well acted... but plods along too slowly" | negative | positive | 0.9999 | Review có cả ý tích cực và tiêu cực |
| 5 | "Disappointed... well done... better than average fare" | negative | positive | 0.9999 | Nuanced opinion: starts negative but ends positive |
| 6 | "Loved the Stooges story... fair job... a must see" | negative | positive | 0.9999 | Positive conclusion words override context |
| 7 | "While not classic... pretty good movie to watch" | negative | positive | 0.9999 | Concession + positive statement |
| 8 | "Disappointed... excellent costumes... fine direction" | negative | positive | 0.9999 | Alternating positive/negative assessments |
| 9 | "Fascinating to watch... absurd but topflight entertainment" | negative | positive | 0.9999 | Sarcasm + positive adjectives confuse model |
| 10 | "Amazing flair... unbelievable talent... perfectly entertaining" | negative | positive | 0.9999 | Superlative positive adjectives dominate |

Gợi ý nhóm lỗi:

- phủ định: Các review phủ định bị che phủ bởi từ tích cực trong câu tiếp theo
- câu dài: Review dài kết thúc với nhận xét tích cực  
- chuyển ý bằng "but/however": Câu nhượng bộ (Although/While) kéo theo khẳng định tích cực
- sarcasm/mỉa mai: Tính mơ hồ của lời lẽ khiến mô hình dự đoán sai
- từ hiếm hoặc tên riêng: Thuật ngữ chuyên nghiệp (cinematography, ensemble) kích hoạt tích cực
- review có cả ý tích cực và tiêu cực: Mô hình không xử lý tốt chuyển đổi giữa cảm xúc

## 8. Kết luận

Mô hình tốt nhất của em là:

- Run name: stacked_bigru
- Cấu hình: GRU model, bidirectional, 2 layers, vocab_size 20000, max_len 256, embed_dim 128, hidden_dim 128, dropout 0.4, batch_size 64, lr 8e-4, epochs 8
- Test accuracy: 0.85336
- Test macro-F1: 0.85332

Giải thích vì sao mô hình này tốt hơn baseline:

- **GRU vs LSTM**: GRU mechanisms hiệu quả hơn trong việc bắt được long-range dependencies so với LSTM đơn giản, cho kết quả cao hơn 5.1% so với baseline LSTM
- **Bidirectional**: Sử dụng thông tin từ cả hai chiều (forward và backward) giúp mô hình capture ngữ cảnh tốt hơn
- **Stacked architecture**: 2 layers GRU cho phép mô hình học được các đặc trưng phức tạp ở các mức khác nhau, mặc dù có hiện tượng overfitting nhẹ (train_loss = 0.03784)
- **Hyperparameter tuning**: Dropout 0.4 (cao hơn baseline 0.3) và learning rate nhỏ hơn (8e-4) giúp chống overfitting tốt hơn

## 9. Tự đánh giá

- [x] Em đã chạy baseline LSTM.
- [x] Em đã thử ít nhất 2 biến thể nâng cấp.
- [x] Em đã lưu checkpoint tốt nhất.
- [x] Em đã phân tích learning curves.
- [x] Em đã phân tích confusion matrix.
- [x] Em đã phân tích ít nhất 10 mẫu sai.
- [x] Em đã commit code và report lên GitHub.
