# 加载最佳模型
from joblib import load
import pandas as pd
from sklearn.model_selection import train_test_split
best_model = load('best_house_price_model.joblib')

# 新数据示例（必须包含所有原始特征，格式与训练数据一致）
predict_data_1 = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housingMedianAge': [41],
    'totalRooms': [880],
    'totalBedrooms': [129],
    'population': [322],
    'households': [126],
    'medianIncome': [8.3252]
})


predicted_price = best_model.predict(predict_data_1)
print(f"数据1预测")
print(f"真实房价: ${452600*100000}")
print(f"预测房价: ${predicted_price[0] * 100000:.2f}")
print(f"\n")

predict_data_2 = pd.DataFrame({
    'longitude': [-122.22],
    'latitude': [37.86],
    'housingMedianAge': [21],
    'totalRooms': [7099],
    'totalBedrooms': [1106],
    'population': [2401],
    'households': [1138],
    'medianIncome': [8.3014]
})

predicted_price = best_model.predict(predict_data_2)
print(f"数据2预测")
print(f"真实房价: ${358500*100000}")
print(f"预测房价: ${predicted_price[0] * 100000:.2f}")
print(f"\n")

predict_data_3 = pd.DataFrame({
    'longitude': [-117.23],
    'latitude': [34.12],
    'housingMedianAge': [6],
    'totalRooms': [4464],
    'totalBedrooms': [1093],
    'population': [2364],
    'households': [952],
    'medianIncome': [2.3848]
})

predicted_price = best_model.predict(predict_data_3)
print(f"数据3预测")
print(f"真实房价: ${81600*100000}")
print(f"预测房价: ${predicted_price[0] * 100000:.2f}")
print(f"\n")


predict_data_4 = pd.DataFrame({
    'longitude': [-121.24],
    'latitude': [39.37],
    'housingMedianAge': [16],
    'totalRooms': [2785],
    'totalBedrooms': [616],
    'population': [1387],
    'households': [530],
    'medianIncome': [2.3886]
})

predicted_price = best_model.predict(predict_data_4)
print(f"数据4预测")
print(f"真实房价: ${89400*100000}")
print(f"预测房价: ${predicted_price[0] * 100000:.2f}")
print(f"\n")
