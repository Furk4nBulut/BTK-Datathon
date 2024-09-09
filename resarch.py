import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Veriyi yükleme
data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test_x.csv')

# Gereksiz verileri düşürme
drop_columns = ['Universite Adi', 'Lise Adi', 'Lise Adi Diger', 'Lise Sehir']
data.drop(drop_columns, axis=1, inplace=True)
test_data.drop(drop_columns, axis=1, inplace=True)

# Kategorik verileri işleme
def extract_city_only(city_string):
    if isinstance(city_string, float):
        return city_string
    city_string = city_string.lower()
    separators = [',', '/', '-', ' ']
    for sep in separators:
        if sep in city_string:
            city_string = city_string.split(sep)[-1].strip()
    return city_string.capitalize()

data['Dogum Yeri'] = data['Dogum Yeri'].apply(extract_city_only)
test_data['Dogum Yeri'] = test_data['Dogum Yeri'].apply(extract_city_only)

def extract_year(date_string):
    try:
        return pd.to_datetime(date_string).year
    except (ValueError, TypeError):
        match = re.search(r'\b(19|20)\d{2}\b', date_string)
        if match:
            return int(match.group(0))
        return None

data['Dogum Tarihi'] = data['Dogum Tarihi'].apply(extract_year)
test_data['Dogum Tarihi'] = test_data['Dogum Tarihi'].apply(extract_year)

# Kategorik ve sayısal özellikleri ayırma
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(data)
test_cat_cols, test_cat_but_car, test_num_cols = grab_col_names(test_data)

# Aykırı değerleri belirleme ve düzeltme
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

for col in num_cols:
    if col != "Degerlendirme Puani":
        print(col, check_outlier(data, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "Degerlendirme Puani":
        replace_with_thresholds(data, col)

# Eksik değerleri kontrol et ve doldur
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_data = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_data, end="\n")
    if na_name:
        return na_columns

missing_values_table(data)

def quick_missing_imp(data, num_method="median", cat_length=20, target="Degerlendirme Puani"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]
    temp_target = data[target]

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target
    return data

data = quick_missing_imp(data, num_method="median", cat_length=20)

# Nadir görülen değerleri 'Rare' olarak işaretleme
def rare_encoder(dataframe, rare_perc):
    temp_data = dataframe.copy()
    rare_columns = [col for col in temp_data.columns if temp_data[col].dtypes == 'O' and (temp_data[col].value_counts() / len(temp_data) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_data[var].value_counts() / len(temp_data)
        rare_labels = tmp[tmp < rare_perc].index
        temp_data[var] = np.where(temp_data[var].isin(rare_labels), 'Rare', temp_data[var])
    return temp_data

data = rare_encoder(data, 0.01)
test_data = rare_encoder(test_data, 0.01)

# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in data.columns if data[col].dtypes == "O" and len(data[col].unique()) == 2]
for col in binary_cols:
    data = label_encoder(data, col)

binary_cols = [col for col in test_data.columns if test_data[col].dtypes == "O" and len(test_data[col].unique()) == 2]
for col in binary_cols:
    test_data = label_encoder(test_data, col)

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

data = one_hot_encoder(data, cat_cols, drop_first=True)
data = one_hot_encoder(data, cat_but_car, drop_first=True)

test_data = one_hot_encoder(test_data, test_cat_cols, drop_first=True)
test_data = one_hot_encoder(test_data, test_cat_but_car, drop_first=True)

# Modelleme ve değerlendirme
y = data["Degerlendirme Puani"]
X = data.drop(["Degerlendirme Puani"], axis=1)

# Eksik değerleri kontrol et ve doldur
y.fillna(y.mean(), inplace=True)

# Modeli tanımla ve eğit
model = LinearRegression()
model.fit(X, y)

# Test veri setinde tahmin yapma
test_data = test_data.reindex(columns=X.columns, fill_value=0)  # Eğitim veri seti ile uyumsuz sütunları eşitle

y_pred = model.predict(test_data)

# Tahminleri ve gerçek değerleri bir DataFrame'e dönüştür
results_df = pd.DataFrame({
    'id': test_data.index,  # Test setindeki indeksleri kullanarak id oluşturun
    'Degerlendirme Puani': y_pred
})

# Tahmin sonuçlarını CSV dosyasına kaydedin
results_df.to_csv('predictions.csv', index=False)

print("Tahminler predictions.csv dosyasına kaydedildi.")

# Performans değerlendirmesi
print("Mean Squared Error:", mean_squared_error(y, model.predict(X)))
print("R^2 Score:", r2_score(y, model.predict(X)))

# Cross-validation ile model değerlendirme
cross_val_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean()}")
