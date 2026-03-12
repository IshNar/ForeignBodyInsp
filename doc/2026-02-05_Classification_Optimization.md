# 2026-02-05 Classification 성능·메모리 최적화 정리

금일 적용한 최적화 방안과 해결 내용을 요약한 문서입니다.

---

## 1. 배경

- **검사 시간 악화**: TensorRT(Level 5) 도입 후 검사 시간이 약 2000ms 수준으로 증가.
- **시간 편차**: Classification(ONNX) 사용 시 180ms~390ms로 편차가 큼. 영상 정지 시에는 상대적으로 안정.
- **메모리 이슈**: 동영상 반복 재생 시 사이클마다 검사 시간이 미세하게 늘어나는 현상.

---

## 2. 원복 사항 (GPU-side crop 시점 기준)

| 항목 | 내용 |
|:---|:---|
| **환경** | PyTorch 2.8.0+cu126 → **2.6.0+cu124** 복구, torch-tensorrt·TensorRT 패키지 제거 |
| **코드** | Level 5(TensorRT) 제거. ONNX는 **항상 CPU pipeline** 사용 (`use_gpu_crop` 조건에서 `ort_session is None`일 때만 GPU crop) |
| **UI** | 별도 DL 다이얼로그 제거 → **RuleBase 파라미터 다이얼로그** 내 **「DL 최적화」 탭**에 Level 0~4 라디오만 복원 |
| **설정** | `rule_params.json`의 `deep_learning.optimization_level` 저장/로드 복원 |

---

## 3. 메모리 관련 최적화

| 방안 | 설명 |
|:---|:---|
| **Worker 버퍼 해제** | `DetectionWorker.run()` 종료 시 `_frame_bgr`, `_full_frame_bgr`를 `None`으로 설정하여 대용량 프레임 참조 해제 |
| **ThreadPoolExecutor 캐싱** | `_run_threshold()`에서 매 프레임 생성하던 2-worker 풀을 `_detect_pool` 멤버 1개로 캐싱 |
| **프레임 복사 축소** | `_on_detection_result`에서 full_frame 4회 copy → 1회만 복사 후 `_display_frame_clean`, `current_frame_bgr`는 참조 공유. `current_frame_bgr`의 추가 `.copy()` 제거 |
| **주기적 GC·CUDA 정리** | `classify_batch`에서 50프레임마다 `gc.collect()` + `torch.cuda.empty_cache()` 호출 |

---

## 4. Classification 구간 시간 변동 완화

| 방안 | 설명 |
|:---|:---|
| **ONNX warmup** | 모델 로드 직후 더미 입력(1, 3, 224, 224)으로 1회 `session.run()` → CUDA workspace 사전 할당, 첫 추론 지연 제거 |
| **GC 구간 격리** | CPU pipeline 추론 구간 진입 시 `gc.disable()`, 종료 시 `gc.enable()` 복원. 추론 중 GC 스파이크로 인한 200ms+ 지연 방지 |
| **ONNX softmax bypass** | logits에서 `argmax` 후 해당 클래스만 이용해 confidence 계산(부분 softmax). 전체 softmax 대비 연산 감소 |
| **ROI 버퍼 풀** | `_extract_rois_chunk(..., out=None)`에 재사용 버퍼 `out` 전달. 유효 ROI만 버퍼 **앞쪽에 연속**으로 채워 반환. 풀 버퍼 2개(`_roi_pool_buffers`)를 두 스레드가 번갈아 사용하여 매 청크 대용량 할당 및 `roi_chw[valid_indices]` 복사 제거 |

---

## 5. 수정 파일

| 파일 | 변경 요약 |
|:---|:---|
| `src/core/classification.py` | Level 5 제거, ONNX warmup, GC 구간 격리, ROI 풀(`_get_roi_pool_buffers`, `_extract_rois_chunk` out·compact), softmax bypass, 50프레임마다 gc/cuda 정리 |
| `src/ui/main_window.py` | Worker 버퍼 해제, `_detect_pool` 캐싱, `_on_detection_result` copy 축소, DL 최적화 탭 연동(dl_classifier 전달, deep_learning 저장/로드) |
| `src/ui/rule_params_dialog.py` | 「DL 최적화」 탭 추가(Level 0~4 라디오), `get_dl_optimization_level`, `_auto_save`/저장·불러오기 시 `deep_learning` 포함 |

---

## 6. 결과 요약

- **일반 구간**: 대부분 0.09~0.12초 대로 안정.
- **동일 프레임(정지)**: 0.03초대까지 감소(캐시·풀 효과).
- **스파이크**: 0.38초대 → 0.24~0.29초 수준으로 완화.
- **메모리**: 사이클당 대용량 복사·할당 감소, Worker·풀 재사용으로 GC 부담 완화.

남는 0.24~0.29초 스파이크는 OS 스케줄링·GPU 동기화 등 외부 요인 영향으로, 코드만으로 제거는 어렵습니다.
