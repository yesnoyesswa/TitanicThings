import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit


script_directory = Path(__file__).parent
TitanicDataset = script_directory / 'Titanic-Dataset.csv'
df = pd.read_csv(TitanicDataset)

    
#Survival rate by Pclass
pclass3_survived_rate = df[(df['Pclass'] == 3)]['Survived'].mean()
pclass2_survived_rate = df[(df['Pclass'] == 2)]['Survived'].mean()
pclass1_survived_rate = df[(df['Pclass'] == 1)]['Survived'].mean()


#Survival rate by Sex
male_survived_rate = df[(df['Sex'] == 'male')]['Survived'].mean()
female_survived_rate = df[(df['Sex'] == 'female')]['Survived'].mean()


#Survival rate by Age
age_group_count = {}
survival_rates = {}
for i in range(0, 85, 5):
    group_data = df[(df['Age'] >= i) & (df['Age'] < i+5)]
    counts = len(group_data)

    if counts > 0:
        rate = group_data['Survived'].mean()
    else:
        rate = 0

    group_name = f"group_{i}_{i+5}"
    age_count = f"age_{i}_{i+5}"

    survival_rates[group_name] = rate
    age_group_count[age_count] = counts

sr_array = np.array(list(survival_rates.values()))
count_array = np.array(list(age_group_count.values()))
age_array = np.arange(2.5, 85, 5)

fake_sr_array = np.array([0.2, 0.17, 0.12, 0.04])
fake_count_array = np.array([300, 300, 300, 1000])
fake_age_array = np.array([85, 90, 95, 100])

sr_array_final = np.concatenate([sr_array, fake_sr_array])
count_array_final = np.concatenate([count_array, fake_count_array])
age_array_final = np.concatenate([age_array, fake_age_array])

sr_array_final = np.clip(sr_array_final, 0, 1)

safe_counts = np.where(count_array_final > 0, count_array_final, 1)
weights_age = 1 / np.sqrt(safe_counts)

def model(x, p7, p6, p5, p4, p3, p2, p1, p0):
    return p7*x**7 + p6*x**6 + p5*x**5 + p4*x**4 + p3*x**3 + p2*x**2 + p1*x + p0

values, _ = curve_fit(model, age_array_final, sr_array_final,
                      sigma = weights_age,
                      absolute_sigma=True, 
                      maxfev=20000)
a_best, b_best, c_best, d_best, e_best, f_best, g_best, h_best = values

import matplotlib.pyplot as plt
import numpy as np

# --- 3. VẼ ĐỒ THỊ ---
plt.figure(figsize=(12, 6))

# Vẽ đường cong dự báo (0 đến 100 tuổi)
x_line = np.linspace(0, 100, 300) # Tăng số điểm lên 300 cho đường cong mượt
y_line = model(x_line, *values)
# Kẹp giá trị trong khoảng [0, 1]
y_line = np.clip(y_line, 0, 1)
plt.plot(x_line, y_line, color='blue', linewidth=3, label='Đường xu hướng (Hàm Bậc 7)')
# Vẽ các điểm dữ liệu (Giữ nguyên như cũ)
plt.scatter(age_array, sr_array, s=count_array*2, color='red', alpha=0.6, label='Dữ liệu thực tế')
mask_fake = fake_age_array <= 100
plt.scatter(fake_age_array[mask_fake], fake_sr_array[mask_fake], 
            s=100, color='green', marker='X', label='Dữ liệu giả lập (Neo)')
# Trang trí
plt.title('Mô hình Tỷ lệ sống sót (Hàm Bậc 7)', fontsize=12)
plt.xlabel('Tuổi')
plt.ylabel('Tỷ lệ sống sót')
plt.xlim(0, 100)
plt.ylim(-0.05, 1.05)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.show()


#Survial rate by SibSp

no0_SibSp_df = df[df['SibSp'] >= 1].copy()

SibSp_count_dict = {}
sr_SibSp = {}

for i in range(1, 4, 1):
    SibSp_data = df[(df['SibSp'] == i)]
    counts = len(SibSp_data)

    rate = SibSp_data['Survived'].mean()

    SibSp_sr = f"SibSp_{i}"
    SibSp_count = f"SibSp_{i}_count"

    sr_SibSp[SibSp_sr] = rate
    SibSp_count_dict[SibSp_count] = counts

SibSp_sr_array = np.array(list(sr_SibSp.values()))
SibSp_count_array = np.array(list(SibSp_count_dict.values()))
SibSp_number_array = np.array([1,2,3])

print(SibSp_count_array)
print(SibSp_sr_array)

def decay_model(x, A, k):
    return A * np.exp(-k * (x - 1))

weights = np.sqrt(SibSp_count_array)

values_SibSp, _ = curve_fit(decay_model, SibSp_number_array, SibSp_sr_array, 
                          p0=[0.55, 0.5], 
                          sigma=1/weights, 
                          absolute_sigma=True)
A_best, k_best = values_SibSp

plt.figure(figsize=(10, 6))
# Vẽ dữ liệu thực tế (Cột)
plt.bar(SibSp_number_array, SibSp_sr_array, color='teal', alpha=0.6, label='Dữ liệu thực (SibSp >= 1)', edgecolor='black')
# Vẽ đường dự báo (Đường cong)
x_line = np.linspace(1, 8, 100) # Chạy từ 1 đến 8
y_line = decay_model(x_line, *values_SibSp)
plt.plot(x_line, y_line, color='red', linewidth=3, label='Mô hình Suy giảm mũ')
plt.title('Mô hình Suy giảm cơ hội sống khi gia đình quá đông', fontsize=14)
plt.xlabel('Số lượng anh chị em/vợ chồng (SibSp)')
plt.ylabel('Tỷ lệ sống sót')
plt.ylim(0, 0.7)
plt.legend()
plt.grid(axis='y', alpha=0.3)    
# Thêm chú thích cho SibSp 0
plt.text(0.5, 0.4, "SibSp 0 (Đi một mình)\nBị loại ra vì\nkhác quy luật", 
            color='gray', style='italic', ha='center')

plt.show()


#Survival rate by Parch
