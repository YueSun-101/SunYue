import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.utils import Bunch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV,cross_val_score,learning_curve
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor,StackingRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 加载数据集
df = pd.read_csv('/root/scikit_learn_data/california_housing/california_housing.csv', header=None)
#print(df.head())
column_names = ['longitude','latitude','housingMedianAge','totalRooms','totalBedrooms','population','households','medianIncome','medianHouseValue']
df.columns = column_names

# 查看数据
print(df.head())
print("\n数据维度:", df.shape)
print("\n统计描述:\n", df.describe())
print("\n缺失值检查:\n", df.isnull().sum())

# 绘制各特征的分布直方图
df.hist(bins=50, figsize=(12, 8))
plt.tight_layout()
plt.savefig('myplot_1.png')

#数据预处理
X = df.drop('medianHouseValue', axis=1)
y = df['medianHouseValue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

total_bedrooms_median = X_train['totalBedrooms'].median()
X_train['totalBedrooms'].fillna(total_bedrooms_median, inplace=True)
X_test['totalBedrooms'].fillna(total_bedrooms_median, inplace=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

#随机森林模型
rf_reg = RandomForestRegressor(n_estimators=100,max_depth=10, random_state=42)
rf_reg.fit(X_train_scaled, y_train)

#XGBoost模型
xgb_reg = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
xgb_reg.fit(X_train_scaled, y_train)

#Stacking集成模型
estimators = [
    ('rf', RandomForestRegressor()),
    ('xgb', XGBRegressor())
]
stack_reg = StackingRegressor(estimators, final_estimator=LinearRegression())
stack_reg.fit(X_train_scaled, y_train)

#模型评估
def evaluate_model(model, X_test, y_test, type):
    y_pred = model.predict(X_test)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    
    # 可视化预测值与真实值对比
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted House Values')
    plt.savefig(f'myplot_{type}.png')

# 评估线性回归
print("Linear Regression:")
evaluate_model(lin_reg, X_test_scaled, y_test, "Linear")

# 评估随机森林
print("\nRandom Forest:")
evaluate_model(rf_reg, X_test_scaled, y_test, "Random")

#评估梯度模型
print("\nXGB Regression:")
evaluate_model(xgb_reg, X_test_scaled, y_test, "XGB")

#评估混合模型
print("\nStack Regression:")
evaluate_model(stack_reg, X_test_scaled, y_test, "Stack")

# 构建预处理pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['medianIncome', 'housingMedianAge', 'totalRooms']),
        ('poly', PolynomialFeatures(degree=2), ['medianIncome', 'totalRooms'])
    ]
)

# 构建完整Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

# 定义参数搜索空间
param_dist = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 4, 5]
}

# 执行优化
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error'
)
search.fit(X_train, y_train)

# 输出最佳模型
best_model = search.best_estimator_
print(f"Best Params: {search.best_params_}")
print(f"Best RMSE: {np.sqrt(-search.best_score_):.2f}")

#保存pipeline
dump(best_model, 'best_house_price_model.joblib')

