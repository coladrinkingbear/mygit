# 환경설정
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import koreanize_matplotlib
pd.set_option('display.float_format', '{:.2f}'.format)

# 데이터 불러오기
r1 = pd.read_csv('건설수주액.csv', index_col=0)
r2 = pd.read_csv('예금은행 대출금리.csv',index_col=0)
r3 = pd.read_csv('건설공사비지수.csv',index_col=0)

# 데이터 전처리
r1=r1.transpose()
r2=r2.transpose()
r3 = r3.drop(r3.columns[0:2], axis=1)
r3 = r3.transpose()
r3 = r3.iloc[:, :1]
r1 = r1.loc['1996.01':]
r2 = r2[["기업대출금리"]]

# 데이터 병합
df = pd.merge(r1, r2, left_index=True, right_index=True)
df = pd.merge(df, r3, left_index=True, right_index=True)
print(df)

# 숫자로 변환
df[['기업대출금리', '건설 통합 공사비지수']] = df[['기업대출금리', '건설 통합 공사비지수']].apply(pd.to_numeric, errors='coerce')

# '건설수주액'을 '건설 통합 공사비지수'로 나눈 값을 새로운 열로 추가
df['보정된 건설수주액'] = df['건설수주액(백만원)']*100 / df['건설 통합 공사비지수']
print(df)

# 학습
X=df['기업대출금리']
y=df['보정된 건설수주액']
X = sm.add_constant(X)
model = sm.OLS(y, X)
result = model.fit()
print(result.summary())

plt.scatter(df['기업대출금리'], df['보정된 건설수주액'], alpha=0.5, label='실제 데이터')
plt.plot(df['기업대출금리'], result.predict(X), color='red', label='회귀선')
plt.title('기업 대출금리와 물가 보정 건설수주액 관계')
plt.xlabel('기업 대출금리')
plt.ylabel('보정된 건설수주액(백만원)')
plt.legend()
plt.show()

# 기업대출금리가 10%일 때의 데이터 생성
new_data = pd.DataFrame({'const': 1, '기업대출금리': 20}, index=[0])

# 학습된 모델을 사용하여 예측
predicted_value = result.predict(new_data)

# 결과 출력
print('기업대출금리가 20%일 때 건설수주액 예측: {:.2f} (백만원)'.format(predicted_value.iloc[0]))