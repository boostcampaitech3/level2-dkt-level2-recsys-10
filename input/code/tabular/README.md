# 동작환경

라이브러리 설치

```
pip install -r requirements.txt
```


# 모델

- lightgbm
- catboost
- lda
- qda
- svc

# 파일

- config.py : WandB sweep 설정 파일
- args.py : 훈련에 필요한 파라미터들을 설정할 수 있는 파일
- tabular/dataloader.py : 데이터의 전처리, Feature Engineering 및 모델에 학습가능한 input 을 만드는 파일 
- tabular/trainer.py : train과 inference에 관련된 함수가 정의된 파일
- tabular/utils.py : 부가 기능 함수가 정의된 파일
- train.py : 시나리오에 따라 데이터를 불러 모델을 학습하는 스크립트
- inference.py : 시나리오에 따라 학습된 모델을 불러 테스트 데이터의 추론값을 계산하는 스크립트
- requirements.txt : 패키지 관리


# 사용 시나리오

- requirements.txt 실행 : 라이브러리 설치(기존 라이브러리 제거 후 설치함)
- config.py 수정 : WandB sweep config 변경
- args.py 수정 : 훈련에 필요한 파라미터들의 설정을 변경
- train.py 실행 : 데이터 학습 수행 및 모델 저장
- inference.py 실행 : 저장된 모델 로드 및 테스트 데이터 추론 수행
