# level2-dkt-level2-recsys-10
<img width="2562" alt="메인로고" src="https://user-images.githubusercontent.com/44939208/169698897-d93de7b1-aff0-43c2-b6e5-8c92037818c3.png">

## ❗ 주제 설명
- 개인 맞춤화 교육을 위해 사용자의 "지식 상태"를 추적하는 딥러닝 방법론인 DKT(Deep Knowledge Tracing) 모델을 구축하고, 사용자가 푼 최종 문제의 정답 여부를 예측

## 📁 데이터
|데이터|데이터설명|
|---|---|
|UserID|사용자의 고유 번호|
|assessmentItemID|사용자가 푼 문항의 일련 번호|
|testId|사용자가 푼 문항이 포함된 시험지의 일련 번호|
|answerCode|사용자가 푼 문항의 정답 여부를 담고 있는 이진(0/1) 데이터 (Target Label)|
|Timestamp|사용자가 문항을 푼 시간 정보|
|KnowledgeTag|사용자가 푼 문항의 고유 태그|
- Iscream-Edu 데이터셋 / CC BY 2.0
- train/test 합쳐서 총 7,442명의 사용자가 존재
- 총 9,454개의 고유 문항, 1,537개의 시험지, 912개의 태그(중분류)가 존재

## 🏆 평가지표
- 주어진 마지막 문제를 맞았는지 틀렸는지로 분류하는 이진 분류 문제이므로, 본 대회에서는 AUROC(주평가지표)와 Accuracy(참고 평가지표)를 사용함


## 👋 팀원 소개

|                                                  [신민철](https://github.com/minchoul2)                                                   |                                                                          [유승태](https://github.com/yst3147)                                                                           |                                                 [이동석](https://github.com/dongseoklee1541)                                                  |                                                                        [이아현](https://github.com/ahyeon0508)                                                                         |                                                                         [임경태](https://github.com/gangtaro)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/52911772?v=4)](https://github.com/minchoul2) | [![Avatar](https://avatars.githubusercontent.com/u/39907669?v=4)](https://github.com/yst3147) | [![Avatar](https://avatars.githubusercontent.com/u/41297473?v=4)](https://github.com/dongseoklee1541) | [![Avatar](https://avatars.githubusercontent.com/u/44939208?v=4)](https://github.com/ahyeon0508) | [![Avatar](https://avatars.githubusercontent.com/u/45648852?v=4)](https://github.com/gangtaro) |

## ⚙ 프로젝트 수행 절차 및 방법
<img width="800" alt="구조도" src="https://user-images.githubusercontent.com/44939208/169698965-c3dc9df1-6d89-48a5-9ffb-a7907aca0692.png">

- Research | 대회와 관련된 기술 탐색 및 학습 진행
- EDA | 주어진 데이터에 대한 파악
- 실험 환경 구축 | 모델 간 성능 비교의 객관화를 위한 실험 환경 구축
- Feature Engineering | 모델의 성능 상승을 위한 새로운 Feature 탐색 및 생성
- Modeling | 목적에 부합하는 모델 선정 및 개발
- 실험 및 평가 | 개발한 모델 또는 새로운 Feature에 대한 성능 평가

## 🔨 Installation & Training
- 각 폴더의 requirements.txt, README.md 참고

## 🏢 Structure
```bash
level2-dkt-level2-recsys-10
├── README.md
├── input
│   └── code
│       ├── dkt
│       │   ├── README.md
│       │   ├── args.py
│       │   ├── config.py
│       │   ├── dkt
│       │   │   ├── criterion.py
│       │   │   ├── dataloader.py
│       │   │   ├── metric.py
│       │   │   ├── model.py
│       │   │   ├── module.py
│       │   │   ├── optimizer.py
│       │   │   ├── scheduler.py
│       │   │   ├── trainer.py
│       │   │   └── utils.py
│       │   ├── inference.py
│       │   ├── requirements.txt
│       │   └── train.py
│       ├── lightgcn
│       │   ├── README.md
│       │   ├── config.py
│       │   ├── inference.py
│       │   ├── lightgcn
│       │   │   ├── datasets.py
│       │   │   ├── models.py
│       │   │   └── utils.py
│       │   ├── requirements.txt
│       │   └── train.py
│       ├── modified_lightgcn
│       │   ├── README.md
│       │   ├── config.py
│       │   ├── inference.py
│       │   ├── lightgcn
│       │   │   ├── datasets.py
│       │   │   ├── models.py
│       │   │   └── utils.py
│       │   ├── requirements.txt
│       │   └── train.py
│       └── tabular
│           ├── README.md
│           ├── args.py
│           ├── config.py
│           ├── fi_split1.png
│           ├── inference.py
│           ├── requirements.txt
│           ├── tabular
│           │   ├── dataloader.py
│           │   ├── trainer.py
│           │   └── utils.py
│           └── train.py
└── references
    ├── modified_lightgcn_embeddingLookUpTableStructure.png
    └── validation_strategy.png
```

## ✨ WRAP-UP REPORT
- [WRAP-UP REPORT](https://poised-speedwell-186.notion.site/P-Stage-3-WRAP-UP-REPORT-8f8b0d9b73654e8dab33b776fd8b5eed)

## 👨‍👩‍👧‍👧 Collaborate Working
- Git Flow 브랜치 전략
<img width="500" height="300" alt="Git Flow" src="https://user-images.githubusercontent.com/44939208/169699327-9c5ccda0-bd2f-46ee-a670-afc02ffea8ea.gif">

- Github Issues 기반 작업 진행
<img width="500" height="300" alt="Git Issues" src="https://user-images.githubusercontent.com/44939208/169699589-0a562d36-9f35-4652-bca7-8ed99e450158.gif">

- Github Projects의 칸반 보드를 통한 일정 관리
<img width="500" height="300" alt="Git Projects" src="https://user-images.githubusercontent.com/44939208/169699978-11fc73d9-3d17-4c19-a081-9a6af680d27e.gif">

- Github Pull requests를 통해 Merge 전 request 검토 & 코드 리뷰
<img width="500" height="300" alt="Git Pull requests" src="https://user-images.githubusercontent.com/44939208/169704384-8ad59779-bbbb-49f9-9ffe-c9b80f3c082e.gif">

- Weights & Biases를 통한 실험 관리
<img width="500" height="300" alt="WandB" src="https://user-images.githubusercontent.com/44939208/169701259-2d285621-a14a-4f7e-ba38-d8c4d12acf46.gif">

- Notion을 활용한 실험 기록 정리
<img width="500" height="300" alt="Notion" src="https://user-images.githubusercontent.com/44939208/169704735-c34b057a-0c28-4a77-a864-253a1ecef375.gif">

## 📜 Reference
- [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart)
- [Deep Knowledge Tracing](https://arxiv.org/pdf/1506.05908.pdf)
- [LONG SHORT-TERM MEMORY](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [BERT: Pre-trainig of Deep Bidirectional Transformers for Language Understanding
](https://arxiv.org/pdf/1810.04805.pdf)
- [Last Query Transformer RNN for Knowledge Tracing](https://arxiv.org/abs/2102.05038)
- [SAINT+: Integrating Temporal Features for EdNet Correctness Prediction](https://arxiv.org/pdf/2010.12042.pdf)
- [Optimizing Deeper Transformers on Small Datasets
](https://arxiv.org/pdf/2012.15355.pdf)
- [LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation](https://arxiv.org/pdf/2002.02126.pdf)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/pdf/1706.09516.pdf)

