

# Heinn-X: Retrospective on Physics-Informed Neural Operator Design

> **"수학적 완벽함이 신경망의 유연함 속에서 어떻게 독이 되는가?"** > 본 프로젝트는 대수적 적분 구조를 Neural Operator에 하드코딩하려는 시도와 그 한계를 수치적으로 규명한 4일간의 연구 기록입니다.

---

## 1. 서론: 가설의 수립 (Hypothesis)

### 🔹 배경
연속 수학(미적분)과 이산 수학(차분/수열합)의 평형 이론인 Heinn-X(H(x) 보정항)를 Neural Operator의 적분 커널에 이식한다면, 데이터에만 의존하지 않는 물리적 무결성(Physical Integrity)을 가진 강력한 AI가 될 것이라 가설을 세웠습니다.

### 🔹 목표
고정된 대수적 적분 행렬(S-Matrix)을 신경망 아키텍처 내부에 직접 구현하여, 이산화 오차(Discretization error)가 존재하지 않는 Perfect PDE Solver를 구축하는 것을 목표로 했습니다.

---

## 2. 실험 과정 및 도메인 확장 (Experimental Phases)

| 단계 | 도메인 | 주요 내용 | 결과 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **다항식 (Polynomials)** | 노이즈 없는 순수 수열의 합 학습 | **성공:** 오차 0에 수렴하는 완벽한 성능 확인 |
| **Phase 2** | **동적 PDE (Dynamic PDE)** | 1D Advection, Burgers' 방정식 적용 | **하락:** 충격파 발생 시 FNO/ChebNO 대비 성능 저하 |
| **Phase 3** | **역문제 및 노이즈** | Poisson 방정식 역산 및 30% 노이즈 테스트 | **실패:** 노이즈가 적분기를 타고 증폭되는 부작용 발생 |

---

## 3. 핵심 발견 (Critical Findings): 왜 터졌는가?

### 수치적 폭주 (Numerical Blow-up)
자기회귀(Autoregressive) 예측 시, $H(x)$ 보정항이 신경망의 미세한 근사 오차를 다항 차수로 증폭시키는 현상을 발견했습니다. 이 오차가 누적되어 경계값에서 값이 무한대로 발산하는 수치적 불안정성을 확인했습니다.

### 경직성(Rigidity)의 함정
하드코딩된 **S-Matrix**는 다항식 세계에선 완벽한 해를 제공하지만, 삼각함수나 지수함수 기반의 실제 물리 데이터에서는 오히려 학습을 방해하는 **'수학적 족쇄'**로 작용함을 데이터로 규명했습니다.
### 노이즈 증폭기 (Noise Amplifier)
적분 연산의 본질인 평활화(Smoothing) 효과를 기대했으나, 엄격한 대수 규칙이 랜덤 노이즈마저 '반드시 적분해야 할 유의미한 신호'로 인식하여 고차원적으로 왜곡하고 증폭시키는 결과를 초래했습니다.

---

## 4. 데이터 기반 최종 결론 (Final Insights)

1. **Heinn-X의 적합 도메인**
   - 노이즈가 없는(Zero-noise) 순수 이산 수학 영역.
   - 기호 추론(Symbolic AI) 및 정수론적 연산자 학습.

2. **Neural Operator 설계 지침**
   - 딥러닝에서 사전 지식(Prior)을 주입할 때는 데이터의 불완전성과 비정형성을 수용할 수 있어야 합니다.
   - **'Hard-constraint'** (수식 하드코딩) 보다는 **'Soft-regularization'** 또는 **'Learnable structure'** 방식의 설계가 필수적임을 확인했습니다.

---

## 5. 프로젝트 소회 및 향후 과제

> *"수학적으로 완벽한 이론이 신경망이라는 유연한 시스템 내에서 어떻게 독이 될 수 있는지 직접 겪어본 값진 경험이었습니다."*

비록 일반적인 PDE Solver로서의 범용성 확보에는 한계를 보였으나, 이 과정에서 Spectral Neural Networks(FNO, ChebNO)의 밑바닥 구현 기술을 확보했습니다. 본 실험을 통해 얻은 물리 데이터 핸들링 능력과 수치 해석적 통찰은 향후 더 유연하고 강력한 **Physics-Informed AI**를 설계하는 데 중요한 밑거름이 될 것입니다.

---

## 🔗 Project Resources
- **Code Repository:** [GitHub - X-NO](https://github.com/Kadoros/X-NO)
- **Developed by:** Hyeon Jegal (Chung-Ang Univ. CVML Lab)

