# 📂 ForeignBodyInsp 문서 (PARA 구조)

> **PARA** = **P**rojects / **A**reas / **R**esources / **A**rchive  
> Tiago Forte의 정보 정리 방법론에 따라 문서를 분류했습니다.

---

## 📁 `1_Projects/` — 프로젝트 (목표 + 기한이 있는 일)

> 구체적인 목표와 마감일이 있는, **진행 중이거나 예정된 프로젝트** 단위의 문서.

| 문서 | 설명 |
|:---|:---|
| `Project_Overview.md` | 프로젝트 전체 개요 (구조, 기능, 실행 방법) |
| `Exhibition_Progress_Report.md` | 전시회 출품 진행 상황 보고서 |
| `Demo_Presentation_Script.md` | 전시회 시연 발표 대본 (7분) |

---

## 📁 `2_Areas/` — 영역 (지속적으로 관리해야 하는 책임 영역)

> 마감일은 없지만, **꾸준히 유지·관리·업데이트해야 하는** 시스템 문서.

| 문서 | 설명 |
|:---|:---|
| `BugReport.md` | 버그 리포트 & 해결 방법 모음 (BUG-001 ~ 015) |
| `Inspection_Pipeline_Flowchart.md` | 전체 검사 파이프라인 플로우차트 (Mermaid + 소스 링크) |
| `Full_Project_Codebase_Walkthrough.md` | 전체 소스코드 해부 가이드 (C++ 개발자용) |
| `Classification_Model_Evolution_History.md` | 분류 모델 변경 이력 (v1 → v2 → v3) |
| `Inspection_Pipeline_Optimization_Guide.md` | 검사 파이프라인 성능 최적화 기록 |

---

## 📁 `3_Resources/` — 리소스 (관심 주제·참고 자료)

> 프로젝트에 직접 귀속되지 않는 **학습 자료, 기술 비교 분석, 설계 근거** 문서.

| 문서 | 설명 |
|:---|:---|
| `Computer_Vision_Deep_Learning_Study_Guide.md` | CV & 딥러닝 완벽 마스터 가이드 (이론+수식+코드) |
| `Bubble_Detection_Algorithm_Comparison.md` | 일반 기포 검출 알고리즘 vs 우리 알고리즘 비교 |
| `EfficientNet_Model_Selection_Rationale.md` | 왜 EfficientNet-B0인가? (모델 선택 근거) |
| `YOLO_vs_Classification_Strategy_Guide.md` | YOLO vs Classification 전략 비교 & 이상 탐지 대안 |
| `Classification_Input_Source_Explanation.md` | 분류기 입력 데이터의 비밀 (원본 컬러 vs 전처리 흑백) |
| `Upgrade_EfficientNet_ONNX_Architecture.md` | ResNet50 → EfficientNet-B0 + ONNX Runtime 전환 가이드 |
| `VSCode_Markdown_Link_Strategy.md` | md 파일에서 소스코드 링크 시 경로 전략 (`/src/...#L줄번호`) |
| `VSCode_Pylance_Import_Resolution.md` | Pylance import 빨간 밑줄 원인과 해결법 (`extraPaths`) |
| `Bubble_Detection_Pipeline_References.md` | 버블 검출 파이프라인 학술적 근거 및 참고 논문 (시연회용) |
| **`__Optical_System_Spec.md`** | 프로젝트에 사용된 카메라(Basler)와 렌즈(Tamron) 및 조명의 광학계 스펙 정리 |
| `Batch_Epoch_Iteration.md` | 딥러닝 기초: Batch Size, Epoch, Iteration 개념 비교 및 정리 |
| `Machine_Vision_Lens_Guide.md` | 머신비전 렌즈 기초 가이드: 단초점 렌즈 vs 배율 고정(텔레센트릭) 렌즈의 차이 |

---

## 📁 `4_Archive/` — 아카이브 (비활성 항목)

> 완료된 프로젝트, 더 이상 사용하지 않는 문서, 과거 버전 자료 등.  
> 현재 비어 있습니다. 프로젝트/영역 문서가 비활성화되면 이곳으로 이동합니다.

---

## 🔍 PARA 분류 기준

| 카테고리 | 핵심 질문 | 특징 |
|:---|:---|:---|
| **Projects** | "이건 언제까지 끝내야 하지?" | 목표 + 기한이 있음 |
| **Areas** | "이건 계속 관리해야 하는 건가?" | 기한 없이 지속적 유지보수 |
| **Resources** | "나중에 참고하면 좋겠는데?" | 관심 주제, 학습 자료, 비교 분석 |
| **Archive** | "이건 더 이상 안 보는 건가?" | 완료/비활성화된 항목 |
