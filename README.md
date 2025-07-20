import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 랜덤포레스트 (분류용)
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files  # 코랩 파일 업로드


print("데이터 파일을 선택해주세요.")
uploaded = files.upload()

# 업로드된 파일 이름 가져오기
for fn in uploaded.keys():
    file_name = fn

# CSV 데이터 로드
df = pd.read_csv(file_name)
print(df.head())

features = ['AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage', 'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee', 'RoamMins']

X = df[features]
y = df['churn']

# 1. 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files  # 코랩 파일 업로드


print("데이터 파일을 선택해주세요.")
uploaded = files.upload()

# 업로드된 파일 이름 가져오기
for fn in uploaded.keys():
    file_name = fn

# CSV 데이터 로드
df = pd.read_csv(file_name)
print(df.head())

# 3. X(입력 변수), y(타겟) 분리
X = df.drop(['Churn'], axis=1)  # 'Churn'을 제외한 모든 컬럼 → 입력 값
y = df['Churn']                # 'Churn' 컬럼만 → 정답 값

# 4. 훈련/테스트 데이터 분리 (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. 랜덤 포레스트 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 예측
y_pred = model.predict(X_test)

# 7. 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f" 정확도(Accuracy): {accuracy:.3f}\n")

print(" [분류 리포트]\n")
print(classification_report(y_test, y_pred))

print(" [혼동 행렬]\n")
print(confusion_matrix(y_test, y_pred))

# 8. Feature Importance 시각화
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from google.colab import files

# 1. 데이터 업로드
print("CSV 파일을 업로드하세요.")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

#  2. 타겟 변수 설정
target_col = 'Churn'  # 실제 타겟 컬럼명으로 바꿔주세요
features = ['AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage', 
            'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 
            'OverageFee', 'RoamMins']

#  3. 이탈 vs 비이탈 평균 비교
mean_diff = df.groupby(target_col)[features].mean().T
print("\n[이탈 vs 비이탈 평균 값]")
print(mean_diff)

#  4. 평균 차이 시각화
plt.figure(figsize=(12,6))
mean_diff.plot(kind='bar', figsize=(12,6))
plt.title('Feature Differences: Churn vs Non-Churn')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.legend(title='Churn')
plt.show()
#  5. 상관관계 히트맵
plt.figure(figsize=(10,6))
sns.heatmap(df[features+[target_col]].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#  6. 각 특징별 Boxplot & Histogram
for col in features:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f'{col} by Churn')
    
    plt.subplot(1,2,2)
    sns.histplot(data=df, x=col, hue=target_col, bins=30, kde=True)
    plt.title(f'{col} Distribution by Churn')
    plt.tight_layout()
    plt.show()

#  7. t-검정 (이탈 vs 비이탈)
churn_group = df[df[target_col]==1]
non_churn_group = df[df[target_col]==0]

results = []
for col in features:
    t_stat, p_val = ttest_ind(churn_group[col], non_churn_group[col], equal_var=False)
    results.append([col, churn_group[col].mean(), non_churn_group[col].mean(), p_val])

#  8. 결과 DataFrame
results_df = pd.DataFrame(results, columns=['Feature','Mean_Churn','Mean_NonChurn','p_value'])
print("\n[특징별 t-test 결과]")
print(results_df)

#  9. CSV로 저장
results_df.to_csv('churn_feature_analysis.csv', index=False)
print("\n 분석 결과 저장 완료: churn_feature_analysis.csv")
