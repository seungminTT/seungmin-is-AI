
# 1. 라이브러리 불러오기

import pandas as pd  #CSV, Excel, SQL 데이터를 사용할떄 유용하다
from sklearn.model_selection import train_test_split # 테스트 모델과 실전 모델를 나눌때 사용한다
from sklearn.feature_extraction.text import CountVectorizer # 단어를 카운트 백터 변환 시키는 친구
from sklearn.linear_model import LogisticRegression # 로지스틱 알고리즘을 부른다
from sklearn.metrics import accuracy_score, classification_report # 알고리즘을 정확도를 알려주는 친구이다
from google.colab import files # 구글 코랩에서 나의 컴퓨터에 있는 파일을 직접 선택하게 해준다


# 2. 데이터 불러오기

print("데이터 파일을 선택해주세요.") # 코랩에서 파일을 업로드 텍스트를 뛰운다
uploaded = files.upload() # 파일 업로드 기능 사용

# 업로드된 파일 이름을 가져옵니다
for fn in uploaded.keys(): 
    uploaded_file_name = fn 

# 업로드된 파일을 DataFrame으로 읽어옵니다
df = pd.read_csv(uploaded_file_name) # pandas로 액셀 데이터를 입력한다
print("데이터 미리보기:")
print(df.head())  # 상위 5행 출력


# 3. 라벨 인코딩 (spam → 1, ham → 0) - 'spam' 컬럼 사용
# Changed 'label' to 'spam' based on the dataframe preview
df['spam'] = df['spam'].map({1: 1, 0: 0}) # Assuming 1 for spam and 0 for ham
# df['spam'] = df['spam'].map({'spam': 1, 'ham': 0}) 원래를 이렇게 하는게 정석이지만 데이터
# 파일에는 spam 만이 라벨이라서 위에 방식대로 썼다

# 4. 입력(X), 타겟(y) 정의 - 'spam' 컬럼 사용

X = df['text']   # 이메일 본문
y = df['spam']  # 스팸 여부 (using 'spam' column)


# 5. 훈련/테스트 데이터 분리

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 6. 텍스트 → 숫자 벡터화 (Bag of Words)

vectorizer = CountVectorizer(stop_words='english')  # 불용어 제거 ex) is ,the ,are 제거 해준다
X_train_vec = vectorizer.fit_transform(X_train)  # 학습 데이터 변환
X_test_vec = vectorizer.transform(X_test)        # 테스트 데이터 변환


# 7. 로지스틱 회귀 모델 학습

model = LogisticRegression(max_iter=1000)  # 반복 횟수 1000으로 설정
model.fit(X_train_vec, y_train) # 모델을 학습시킨다 


# 8. 테스트 데이터 예측

y_pred = model.predict(X_test_vec) 

# 정확도 출력
print("\n 모델 정확도:", accuracy_score(y_test, y_pred))

# 정밀도, 재현율, F1-score 출력
print("\n[ 분류 리포트]\n", classification_report(y_test, y_pred))


# 9. 새로운 이메일 예측 예시

sample_email = ["Congratulations! You've won a free ticket, click here"]
sample_email_vec = vectorizer.transform(sample_email)

# 예측 결과 (0: 정상, 1: 스팸)
prediction = model.predict(sample_email_vec)
probability = model.predict_proba(sample_email_vec)

print("\n 새로운 이메일 예측 결과:")
print("예측 라벨 (0: 정상, 1: 스팸):", prediction[0])
print(f"스팸 확률: {probability[0][1]:.2f}, 정상 확률: {probability[0][0]:.2f}")