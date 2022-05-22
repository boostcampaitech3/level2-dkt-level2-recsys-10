# 동작환경

라이브러리 설치

```
pip install -r requirements.txt
```


# 모델

- LSTM
- LSTMATTN
- Bert
- Saint
- LastQuery
- FixupEncoder

# 파일

- config.py : WandB sweep 설정 파일
- args.py : 훈련에 필요한 파라미터들을 설정할 수 있는 파일
- dkt/criterion.py : loss함수가 정의된 파일(BCELoss)
- dkt/dataloader.py : 데이터의 전처리 및 모델에 학습가능한 input 을 만드는 파일
- dkt/metric.py : 평가 지표가 정의된 파일(roc_auc, accuracy)
- dkt/model.py : 모델이 포함된 파일
- dkt/module.py : 모델에 필요한 Encoding, EncoderLayer 등이 정의된 파일
- dkt/optimizer.py : 훈련에 사용될 optimizer가 정의된 파일(Adam, AdamW)
- dkt/scheduler.py : learing rate을 조절하기 위한 scheduler가 포함된 파일(ReduceLROnPlateau, getlinearschedulewithwarmup)
- dkt/trainer.py : train, validation 등과 같은 실제로 모델이 훈련이 되는 로직이 포함된 파일
- dkt/utils.py : 부가 기능 함수가 정의된 파일
- train.py : 메인 파일로 훈련을 시작할 때 사용되는 파일
- inference.py : 학습이 완료된 모델을 이용해 test 데이터를 기반으로 예측된 csv파일을 생성하는 파일
- requirements.txt : 패키지 관리


# 사용 시나리오

- requirements.txt 실행 : 라이브러리 설치(기존 라이브러리 제거 후 설치함)
- config.py 수정 : WandB sweep config 변경
- args.py 수정 : 훈련에 필요한 파라미터들의 설정을 변경
- train.py 실행 : 데이터 학습 수행 및 모델 저장
- inference.py 실행 : 저장된 모델 로드 및 테스트 데이터 추론 수행