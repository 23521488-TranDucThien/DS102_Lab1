import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv(r"C:\Users\Admin\Documents\Đại học\DS102\Thực Hành\BTTH1\forest+fires\forestfires.csv")
df

df.info()

# Hàm chuyển tháng sang số
def convert_month(month: str) -> int:
    month_dict = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }
    return month_dict[month]

def convert_day(day: str) -> int:
    day_dict = {
        'sun': 0,
        'mon': 1, 
        'tue': 2,
        'wed': 3,
        'thu': 4,
        'fri': 5,
        'sat': 6,
    }
    return day_dict[day]

# Chuyển chữ sang số
df['day'] = df['day'].apply(convert_day)

# Chuyển chữ sang số
df['month'] = df['month'].apply(convert_month)

df

class LinearRegression:
    # Hàm huấn luyện mô hình hồi quy tuyến tính.
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        cov = X.T @ X
        inv_cov = np.linalg.inv(cov)
        self.theta_: np.ndarray = inv_cov @ (X.T @ y)
    
    # Hàm tính toán sai số trung bình bình phương căn gốc (Root Mean Square Error).
    def rmse(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        delta = y - y_hat
        
        return (delta**2).mean()**0.5
    
    # Hàm dự đoán các giá trị đầu ra dựa trên dữ liệu đầu vào X.
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.matmul(self.theta_.T, X.T)
        
        return y_pred
    
lr = LinearRegression()

X_y = df.to_numpy()

X_y.shape

N = df.shape[0]
X_y_train, X_y_test = np.split(X_y, indices_or_sections = [int(0.8*N)])

X_train, X_test = X_y_train[:, :-1], X_y_test[:, :-1]
X_train.shape, X_test.shape

y_train, y_test = X_y_train[:, -1], X_y_test[:, -1]
y_train.shape, y_test.shape

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

lr.rmse(y_pred, y_test)