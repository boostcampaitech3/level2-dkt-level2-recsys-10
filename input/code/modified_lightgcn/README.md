# 동작환경

라이브러리 설치

```
pip install -r requirements.txt
```


# 파일

- config.py : 설정 파일
- lightgcn/datasets.py : 데이터 로드 및 전처리 함수 정의
- lightgcn/model.py : 모델을 정의하고 manipulation 하는 build, train, inference관련 코어 로직 정의
- lightgcn/utils.py : 부가 기능 함수 정의
- train.py : 시나리오에 따라 데이터를 불러 모델을 학습하는 스크립트
- inference.py : 시나리오에 따라 학습된 모델을 불러 테스트 데이터의 추론값을 계산하는 스크립트
- requirements.txt : 패키지 관리


# 사용 시나리오

- requirements.txt 실행 : 라이브러리 설치(기존 라이브러리 제거 후 설치함)
- config.py 수정 : 데이터 파일/출력 파일 경로 설정 등
- train.py 실행 : 데이터 학습 수행 및 모델 저장
- inference.py 실행 : 저장된 모델 로드 및 테스트 데이터 추론 수행
