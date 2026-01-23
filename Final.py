import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

script_directory = Path.cwd() 
TitanicDataset = script_directory / 'Titanic-Dataset.csv'
df = pd.read_csv(TitanicDataset)



# --- LOGIC PER-COLUMN ---

df_clone = df.copy()

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
    #Phân cụm tuổi
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

#Dữ liệu thực tế
sr_array = np.array(list(survival_rates.values()))
count_array = np.array(list(age_group_count.values()))
age_array = np.arange(2.5, 85, 5)

#Neo kiến thức (dữ liệu giả lập)
fake_sr_array = np.array([0.2, 0.17, 0.12, 0.04])
fake_count_array = np.array([300, 300, 300, 1000])
fake_age_array = np.array([85, 90, 95, 100])

#Dữ liệu tổng hợp
sr_array_final = np.concatenate([sr_array, fake_sr_array])
count_array_final = np.concatenate([count_array, fake_count_array])
age_array_final = np.concatenate([age_array, fake_age_array])

#Chặn 2 đầu tỉ lệ sống
sr_array_final = np.clip(sr_array_final, 0, 1)

safe_counts = np.where(count_array_final > 0, count_array_final, 1)
#Tính trọng số
weights_age = 1 / np.sqrt(safe_counts)

#Hàm bậc 7
def model(x, p7, p6, p5, p4, p3, p2, p1, p0):
    return p7*x**7 + p6*x**6 + p5*x**5 + p4*x**4 + p3*x**3 + p2*x**2 + p1*x + p0

#Curve_fit
age_values, _ = curve_fit(model, age_array_final, sr_array_final,
                      sigma = weights_age,
                      absolute_sigma=True, 
                      maxfev=20000)


#Survial rate by SibSp

no0_SibSp_df = df[df['SibSp'] >= 1].copy()

full_SibSp_count_dict = {}
full_sr_SibSp_dict = {}

for i in range(0, 4, 1):
    full_SibSp_data = df[(df['SibSp'] == i)]
    full_counts = len(full_SibSp_data)

    full_rate = full_SibSp_data['Survived'].mean()

    full_SibSp_sr = f"SibSp_{i}"
    full_SibSp_count = f"SibSp_{i}_count"

    full_sr_SibSp_dict[full_SibSp_sr] = full_rate
    full_SibSp_count_dict[full_SibSp_count] = full_counts

full_SibSp_sr_array = np.array(list(full_sr_SibSp_dict.values()))
full_SibSp_count_array = np.array(list(full_SibSp_count_dict.values()))
full_SibSp_number_array = np.array([0,1,2,3])


SibSp_count_dict = {}
sr_SibSp_dict = {}

for i in range(1, 4, 1):
    SibSp_data = df[(df['SibSp'] == i)]
    counts = len(SibSp_data)

    rate = SibSp_data['Survived'].mean()

    SibSp_sr = f"SibSp_{i}"
    SibSp_count = f"SibSp_{i}_count"

    sr_SibSp_dict[SibSp_sr] = rate
    SibSp_count_dict[SibSp_count] = counts

SibSp_sr_array = np.array(list(sr_SibSp_dict.values()))
SibSp_count_array = np.array(list(SibSp_count_dict.values()))
SibSp_number_array = np.array([1,2,3])

def decay_model(x, A, k):
    return A * np.exp(-k * (x - 1))

weights = np.sqrt(SibSp_count_array)

values_SibSp, _ = curve_fit(decay_model, SibSp_number_array, SibSp_sr_array, 
                          p0=[0.55, 0.5], 
                          sigma=1/weights, 
                          absolute_sigma=True)


#Survival rate by Parch

Parch_count_dict = {}
sr_Parch_dict = {}

#print(df['Parch'].max()) -> 6

for i in range(0, 7, 1):
    Parch_data = df[(df['Parch'] == i)]
    counts = len(Parch_data)

    rate = Parch_data['Survived'].mean()

    Parch_sr = f"SibSp_{i}"
    Parch_count = f"SibSp_{i}_count"

    sr_Parch_dict[Parch_sr] = rate
    Parch_count_dict[Parch_count] = counts

def gaussian_func(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

Parch_sr_array = np.array(list(sr_Parch_dict.values()))
Parch_count_array = np.array(list(Parch_count_dict.values()))
Parch_number_array = np.array([0,1,2,3,4,5,6])

values_parch, _ = curve_fit(gaussian_func, Parch_number_array, Parch_sr_array,
                             p0=[0.6, 1.5, 1.0])


#Survival rate by Ticket
df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')

ticket_freq_rates = {}
ticket_freq_counts = {}

max_freq = int(df['Ticket_Frequency'].max())

for i in range(1, max_freq + 1):
    group_data = df[df['Ticket_Frequency'] == i]
    if len(group_data) > 0:
        ticket_freq_rates[i] = group_data['Survived'].mean()
        ticket_freq_counts[i] = len(group_data)

x_ticket = np.array(list(ticket_freq_rates.keys()))
y_ticket = np.array(list(ticket_freq_rates.values()))
counts_ticket = np.array(list(ticket_freq_counts.values()))

def gaussian_model(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

weights_ticket = 1 / np.sqrt(counts_ticket)

values_ticket, _ = curve_fit(gaussian_model, x_ticket, y_ticket, 
                    p0=[0.6, 2.5, 1.5], 
                    sigma=weights_ticket)


#Survival rate by Fare

fare_group_count = {}
fare_survival_rates = {}

# Phân cụm
max_fare = int(df['Fare'].max())
for i in range(0, max_fare, 20):
    group_data = df[(df['Fare'] >= i) & (df['Fare'] < i+20)]
    counts = len(group_data)

    if counts > 0:
        rate = group_data['Survived'].mean()
    else:
        rate = 0

    group_name = f"fare_group_{i}_{i+20}"
    count_name = f"fare_count_{i}_{i+20}"

    fare_survival_rates[group_name] = rate
    fare_group_count[count_name] = counts

sr_fare_final = np.array(list(fare_survival_rates.values()))
count_fare_final = np.array(list(fare_group_count.values()))

fare_array_final = np.arange(10, max_fare, 20) # Lấy trung điểm của mỗi cụm

sr_fare_final = np.clip(sr_fare_final, 0, 1) # Chặn tỉ lệ sống trong khoảng [0, 1]

safe_counts = np.where(count_fare_final > 0, count_fare_final, 1)
weights_fare = 1 / np.sqrt(safe_counts)

#Hàm bão hòa
def saturation_model(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

values_fare, _ = curve_fit(saturation_model, fare_array_final, sr_fare_final,
                           sigma = weights_fare,
                           absolute_sigma=True, 
                           maxfev=20000,
                           p0=[0.8, 0.05, 40])


#Survival rate by Embarked
e_S_survived_rate = df[(df['Embarked'] == 'S')]['Survived'].mean()
e_C_survived_rate = df[(df['Embarked'] == 'C')]['Survived'].mean()
e_Q_survived_rate = df[(df['Embarked'] == 'Q')]['Survived'].mean()


age_values = np.array(age_values)
values_SibSp = np.array(values_SibSp)
values_parch = np.array(values_parch)
values_ticket = np.array(values_ticket)
values_fare = np.array(values_fare)

values_pclass = np.array([pclass1_survived_rate, pclass2_survived_rate, pclass3_survived_rate])
values_sex = np.array([male_survived_rate, female_survived_rate])
values_embarked = np.array([e_S_survived_rate, e_C_survived_rate, e_Q_survived_rate])


#Các cột theo phương trình
def predict_survival_age(age, p):
    #p: Mảng các hệ số [p7, p6, ..., p0] thu được từ curve_fit
    sr = np.polyval(p, age) # np.polyval tính toán: p[0]*x^n + p[1]*x^(n-1) + ... + p[n]

    return np.clip(sr, 0, 1)

def predict_survival_sibsp(sibsp, v):
    if sibsp == 0:
        return 0.345
    # v = [A, k]
    sr = v[0] * np.exp(-v[1] * (sibsp - 1))
    return np.clip(sr, 0, 1)

def predict_survival_parch(parch, v):
    # v = [a, mu, sigma]
    sr = v[0] * np.exp(-(parch - v[1])**2 / (2 * v[2]**2))

    return np.clip(sr, 0, 1)

def predict_survival_ticket_freq(freq, v):
    # v = [a, mu, sigma]
    sr = v[0] * np.exp(-(freq - v[1])**2 / (2 * v[2]**2))
    return np.clip(sr, 0, 1)

def predict_survival_fare(fare, v):
    # v = [L, k, x0]
    res = v[0] / (1 + np.exp(-v[1] * (fare - v[2])))
    return np.clip(res, 0, 1)

#Các cột theo hằng số
def predict_survival_pclass(pclass, rate):
    idx = int(pclass) - 1  #Chuyển 1,2,3 thành index 0,1,2
    return rate[idx]

def predict_survival_sex(sex, rate):
    if sex.lower() == 'female': 
        idx = 1
    else:
        idx = 0
    return rate[idx]

def predict_survival_embarked(embarked, rate):
    converter = {'S': 0, 'C': 1, 'Q': 2}
    idx = converter.get(embarked.upper(), 0) # Mặc định là S nếu không khớp
    return rate[idx]



# --- LOGIC OVERALL ---


df = df_clone


# Xóa PassengerId:
df = df.drop(columns=['PassengerId'])

# Tách cột Survived ra riêng:
df_survived = df['Survived'].values
df = df.drop(columns=['Survived'])

# Xử lý cột Pclass: không cần

# Xóa Name:
df = df.drop(columns=['Name'])

# Xử lý cột Sex:
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Xử lý cột Age: Điền các giá trị còn thiếu bằng số Trung vị (28.0)
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

# Xử lý cột SibSp: không cần

# Xử lý cột Parch: không cần

# Xử lý cột Ticket: 
    # Đếm số lần xuất hiện của mỗi mã vé 
ticket_counts = df['Ticket'].value_counts()
df['TicketFrequency'] = df['Ticket'].map(ticket_counts)
    # Xóa cột Ticket cũ
df = df.drop(columns=['Ticket'])

# Xử lý Fare: không cần

# Xóa Cabin(quá ít dữ liệu): 
df = df.drop(columns=['Cabin'])

# Xử lý Embarked: C (Cherbourg), Q (Queenstown), và S (Southampton)
df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
# Cả 2 cột bằng 0 thì tức là Embarked = S
df = df.drop(columns=['Embarked'])

# Thêm cột Bias:
df['Bias'] = 1

#Các cột cần chuẩn hóa
std_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'TicketFrequency', 'Fare']

df_processed = df.copy()
df_before = [df[i] for i in std_cols]
df_after = []

#Chuẩn hóa: z = (x - mean) / standard deviation
for i in std_cols:
    mean_val = df[i].mean() #trung bình
    std_val = df[i].std() #độ lệch chuẩn
    df[i] = (df[i] - mean_val) / std_val
    df_after.append(df[i])

#Viết lại df thành X:
X = df
#Viết lại Survived thành y:
y = df_survived

#Tính ma trận chuyển vị của X
XT = X.T
#tính X^T * X
XTX = XT.dot(X)
#Tính X^T * y 
XTy = XT.dot(y)

#Tính nghịch đảo của XTX
XTX_inv = np.linalg.inv(XTX)

#Tính weights
w = XTX_inv.dot(XTy)

weights = np.array(w)

# Tạo dict để lưu thông số
stats = {}

# Tính Mean
for i in std_cols:
    stats[i] = {'mean': df_processed[i].mean()}

# Tính Std
for i in std_cols:
    stats[i]['std'] = df_processed[i].std()

# In dict stats
for col, val in stats.items():
    print(f"{col}: Mean = {val['mean']}, Std = {val['std']}")

# Hàm tính xác suất sống sót qua thông tin đầu vào (nếu thiếu thông tin thì trả về giá trị mặc định)
def predict_survival(pclass=3, sex='male', age=28.0, sibsp=0, parch=0, 
                     ticket_freq=1, fare=14.45, embarked='S'):
    
    # Chuẩn hóa
    pclass_std = (pclass - stats['Pclass']['mean']) / stats['Pclass']['std']
    age_std = (age - stats['Age']['mean']) / stats['Age']['std']
    sibsp_std = (sibsp - stats['SibSp']['mean']) / stats['SibSp']['std']
    parch_std = (parch - stats['Parch']['mean']) / stats['Parch']['std']
    ticketfreq_std = (ticket_freq - stats['TicketFrequency']['mean']) / stats['TicketFrequency']['std']
    fare_std = (fare - stats['Fare']['mean']) / stats['Fare']['std']
    
    # Xử lý sex và embarked
    sex_num = 1 if sex.lower() == 'female' else 0
    embarked_c = 1 if embarked.upper() == 'C' else 0
    embarked_q = 1 if embarked.upper() == 'Q' else 0
    
    # Vector input (Khớp đúng thứ tự Weight)
    vector_input = np.array([pclass_std, sex_num, age_std, sibsp_std, parch_std, ticketfreq_std, fare_std, embarked_c, embarked_q, 1.0])
    
    # Tính toán
    sr = np.dot(vector_input, weights)
    return sr

def get_infor_overall():
    print("\n--- NHAP THONG TIN HANH KHACH ---")
    
    # Pclass
    p = input("Nhap Hang ve (1, 2, 3) [Mac dinh 3]: ")
    if p == "1" or p == "2" or p == "3":
        pclass = float(p)
    else:
        pclass = 3.0 # Mac dinh la 3

    # Sex
    s = input("Nhap Gioi tinh (male/female) [Mac dinh male]: ").lower() # Cho cả chữ viết hoa hợp lệ
    if s == "female":
        sex = "female"
    else:
        sex = "male" # Mac dinh la nam

    # Age
    a = input("Tuoi (0-100) [Mac dinh 28]: ")
    if a.replace('.', '', 1).isdigit():  # Số chỉ hợp lệ nếu có 1 dấu "."
        age = float(a)
        if age < 0 or age > 100:
            age = 28.0
    else:
        age = 28.0

    # Fare
    f = input("Nhap Gia ve [Mac dinh 14.45]: ")
    if f.replace('.', '', 1).isdigit():
        fare = float(f)
    else:
        fare = 14.45

    # Embarked
    e = input("Nhap Cang (S, C, Q) [Mac dinh S]: ").upper() # Cho cả chữ viết thường hợp lệ
    if e == "C" or e == "Q" or e == "S":
        embarked = e
    else:
        embarked = "S"

    # SipSp
    sib = input("So anh chi em/ vo chong di cung [Mac dinh 0]: ")
    if sib.isdigit(): 
        sibsp = float(sib)
        if sibsp < 0: 
            sibsp = 0.0
    else:
        sibsp = 0.0

    # Parch
    par = input("So bo me/ con cai di cung [Mac dinh 0]: ")
    if par.isdigit():
        parch = float(par)
        if parch < 0: 
            parch = 0.0
    else:
        parch = 0.0

    # Ticket Frequency 
    tf = input("So nguoi dung chung 1 ma ve [Mac dinh 1]: ")
    if tf.isdigit():
        ticket_freq = float(tf)
        if ticket_freq < 1: 
            ticket_freq = 1.0
    else:
        ticket_freq = 1.0

    # Lưu tất cả vào Dict
    data = {
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'fare': fare,
        'embarked': embarked,
        'sibsp': sibsp,
        'parch': parch,
        'ticket_freq': ticket_freq
    }
    
    return data



# --- GIAO DIỆN CHƯƠNG TRÌNH ---


def menu_per_column():
    while True:
        print("\n" + "-"*15 + " DỰ ĐOÁN THEO CỘT (LỌC TRỰC TIẾP) " + "-"*15)
        print("1. Kiểm tra theo Tuổi (Age)")
        print("2. Kiểm tra theo Anh chị em/Vợ chồng (SibSp)")
        print("3. Kiểm tra theo Bố mẹ/Con cái (Parch)")
        print("4. Kiểm tra theo Giá vé (Fare)")
        print("0. Quay lại Menu chính")
        
        choice = input("\nChọn cột (0-4): ")
        
        # --- TUỔI ---
        if choice == '1':
            while True:
                val_in = input("Nhập Tuổi (0-100): ").strip()
                # Kiểm tra xem có phải là số (có thể có 1 dấu chấm thập phân)
                if val_in.replace('.', '', 1).isdigit():
                    val = float(val_in)
                    if 0 <= val <= 100:
                        res = predict_survival_age(val, age_values)
                        print(f"==> Tỉ lệ sống sót cho người {val} tuổi: {res*100:.2f}%")
                        break # Thoát vòng lặp nhập liệu khi hợp lệ
                    else:
                        print("X Lỗi: Vui lòng nhập tuổi trong khoảng 0-100.")
                else:
                    print("X Lỗi: Định dạng không hợp lệ, hãy nhập số.")

        # --- SIBSP ---
        elif choice == '2':
            while True:
                val_in = input("Nhập số SibSp (0-8): ").strip()
                if val_in.isdigit(): # SibSp là số nguyên
                    val = int(val_in)
                    if 0 <= val <= 8:
                        res = predict_survival_sibsp(val, values_SibSp)
                        print(f"==> Tỉ lệ sống sót khi có {val} SibSp: {res*100:.2f}%")
                        break
                    else:
                        print("X Lỗi: SibSp thường từ 0 đến 8.")
                else:
                    print("X Lỗi: Vui lòng nhập số nguyên.")

        # --- PARCH ---
        elif choice == '3':
            while True:
                val_in = input("Nhập số Parch (0-6): ").strip()
                if val_in.isdigit():
                    val = int(val_in)
                    if 0 <= val <= 6:
                        res = predict_survival_parch(val, values_parch)
                        print(f"==> Tỉ lệ sống sót khi có {val} Parch: {res*100:.2f}%")
                        break
                    else:
                        print("X Lỗi: Parch thường từ 0 đến 6.")
                else:
                    print("X Lỗi: Vui lòng nhập số nguyên.")

        # --- FARE ---
        elif choice == '4':
            while True:
                val_in = input("Nhập Giá vé (0-600): ").strip()
                if val_in.replace('.', '', 1).isdigit():
                    val = float(val_in)
                    if 0 <= val <= 600:
                        res = predict_survival_fare(val, values_fare)
                        print(f"==> Tỉ lệ sống sót với giá vé {val}$: {res*100:.2f}%")
                        break
                    else:
                        print("X Lỗi: Giá vé không hợp lệ (0-600).")
                else:
                    print("X Lỗi: Vui lòng nhập số tiền.")

        elif choice == '0':
            break
        else:
            print("X Lựa chọn menu không hợp lệ!")

def main_menu():
    while True:
        print("\n" + "="*40)
        print("   HỆ THỐNG DỰ ĐOÁN TITANIC (OVERALL)")
        print("="*40)
        print("1. Dự đoán theo từng đặc tính (Column Logic)")
        print("2. Dự đoán tổng quát (Nhập đầy đủ thông tin)")
        print("0. Thoát")
        print("="*40)
        
        main_choice = input("Chọn chức năng: ")
        
        if main_choice == '1':
            menu_per_column()
            
        elif main_choice == '2':
            print("\n--- NHẬP THÔNG TIN CHI TIẾT ---")
            
            # Lọc Pclass
            while True:
                p_in = input("- Hạng vé (1, 2, 3): ").strip()
                if p_in in ['1', '2', '3']:
                    pclass = float(p_in)
                    break
                else:
                    print("X Lỗi: Chỉ nhập 1, 2 hoặc 3.")

            # Lọc Sex
            while True:
                s_in = input("- Giới tính (male/female): ").lower().strip()
                if s_in in ['male', 'female']:
                    sex = s_in
                    break
                else:
                    print("X Lỗi: Vui lòng nhập đúng 'male' hoặc 'female'.")

            # Lọc Age
            while True:
                a_in = input("- Tuổi (0-100): ").strip()
                if a_in.replace('.', '', 1).isdigit():
                    age = float(a_in)
                    if 0 <= age <= 100:
                        break
                    else:
                        print("X Lỗi: Tuổi từ 0 đến 100.")
                else:
                    print("X Lỗi: Vui lòng nhập số.")

            # Lọc SibSp
            while True:
                sib_in = input("- Số anh chị em/vợ chồng đi cùng (0-8): ").strip()
                if sib_in.isdigit():
                    sibsp = float(sib_in)
                    if 0 <= sibsp <= 8:
                        break
                    else:
                        print("X Lỗi: Nhập trong khoảng 0-8.")
                else:
                    print("X Lỗi: Nhập số nguyên.")

            # Lọc Parch
            while True:
                par_in = input("- Số bố mẹ/con cái đi cùng (0-6): ").strip()
                if par_in.isdigit():
                    parch = float(par_in)
                    if 0 <= parch <= 6:
                        break
                    else:
                        print("X Lỗi: Nhập trong khoảng 0-6.")
                else:
                    print("X Lỗi: Nhập số nguyên.")

            # Lọc Ticket Frequency
            while True:
                tf_in = input("- Số người dùng chung mã vé (>= 1): ").strip()
                if tf_in.isdigit():
                    ticket_freq = float(tf_in)
                    if ticket_freq >= 1:
                        break
                    else:
                        print("X Lỗi: Ít nhất là 1 người.")
                else:
                    print("X Lỗi: Nhập số nguyên.")

            # Lọc Fare
            while True:
                f_in = input("- Giá vé (0-600): ").strip()
                if f_in.replace('.', '', 1).isdigit():
                    fare = float(f_in)
                    if 0 <= fare <= 600:
                        break
                    else:
                        print("X Lỗi: Giá vé từ 0 đến 600.")
                else:
                    print("X Lỗi: Nhập số tiền.")

            # Lọc Embarked
            while True:
                e_in = input("- Cảng lên tàu (S, C, Q): ").upper().strip()
                if e_in in ['S', 'C', 'Q']:
                    embarked = e_in
                    break
                else:
                    print("X Lỗi: Chỉ nhập S, C hoặc Q.")

            # --- DỰ ĐOÁN ---
            prob = predict_survival(pclass, sex, age, sibsp, parch, ticket_freq, fare, embarked)
            
            prob_display = np.clip(prob, 0, 1) * 100
            print("\n" + "*"*35)
            print(f"KẾT QUẢ PHÂN TÍCH TỔNG HỢP:")
            print(f"Xác suất sống sót: {prob_display:.2f}%")
            print("*"*35)
            
            input("\nNhấn Enter để quay lại menu chính...")

        elif main_choice == '0':
            print("Đang thoát chương trình. Tạm biệt!")
            break
        else:
            print("X Lựa chọn không hợp lệ!")

main_menu()