from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pandas.read_csv(r"C:\Users\ballc\OneDrive\Documents\Desktop\data analytics final\data.csv")


print(f"[!] Size of dataset before dropping duplicate/null values: {len(df.axes[0])}")
df = df.drop_duplicates()
print(f"[!] Size of dataset after dropping duplicate/null values: {len(df.axes[0])}")

print(df['SalePrice'].describe())


unwanted_features = ['Condition1', 'Condition2', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageYrBlt',
                        'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'GarageCars', 'MSZoning', 'OverallQual']

df = df.drop(columns = unwanted_features)

print(f"[!] Number of features left after dropping unwanted columns: {len(df.dtypes)}")

numeric_cols = ['GrLivArea', 'LotFrontage', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'MasVnrArea', 'GarageArea', 'LotArea']
# Create subplots for boxplots
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(20, 6))

# Plot boxplots for each column
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()
#q1 = .25
#q2 = .50
#q3 = .75
#q4 = 1.00
q1 = df[numeric_cols].quantile(0.25)
q3 = df[numeric_cols].quantile(0.75)
iqr = q3 - q1

#define outlier thresholds
lower_bound = q1 - 3 * iqr
upper_bound = q3 + 3 * iqr

#create a boolean mask for rows without outliers
mask = (df[numeric_cols] >= lower_bound) & (df[numeric_cols] <= upper_bound)

#keep only rows without outliers
df = df[mask.all(axis = 1)]

print(f"[!] Size of df without outliers: {len(df.axes[0])}")

num_plots = len(numeric_cols)
cols = 3
rows = (num_plots // cols) + (num_plots % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))

axes = axes.flatten()

#plot scatterplots
for i, col in enumerate(numeric_cols):
    sns.scatterplot(x=df[col], y=df['SalePrice'], ax=axes[i])
    axes[i].set_title(f'SalePrice vs {col}', pad = 5)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('SalePrice')
    axes[i].tick_params(axis = 'x', rotation = 45, labelsize = 10) #rotate x-axis labels and adjust size

# Remove empty subplots if any
for i in range(num_plots, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Increase horizontal and vertical space between plots

plt.show()

categoricals = ['LotConfig', 'PavedDrive', 'KitchenQual', 'BsmtQual', 'LandContour', 'BldgType', 'Neighborhood', 'GarageType', 'CentralAir', 'HeatingQC', 'Heating', 'GarageFinish', 'Exterior2nd', 'HouseStyle']

for category in categoricals:
    uniques = df[category].unique()
    mapping = {value: i for i, value in enumerate(uniques)}
    df[category] = df[category].map(mapping)


numeric_cols = numeric_cols + categoricals
print(numeric_cols)
print(df.head())

x = df[categoricals]
y = df['SalePrice']

#ANOVA
f_values, p_values = f_classif(x, y)

to_drop = []

for x in range(len(p_values)):
    print(f"[!] {categoricals[x]} Statistics")
    print(f'ANOVA F-values: {f_values[x]}')
    print(f'ANOVA p-values: {p_values[x]}')
    if p_values[x] > .05:
        print(f'[!] {categoricals[x]} is not significant!')
        to_drop.append(categoricals[x])

for drop in to_drop:
    numeric_cols.remove(drop)


numeric_cols.remove("MasVnrArea")

x_significant = df[numeric_cols]
print(numeric_cols)
y = df['SalePrice']

#Question 4

x_train, x_test, y_train, y_test = train_test_split(x_significant, y, test_size = 0.2, random_state = 42)

model = XGBRegressor()
model.fit(x_train, y_train)

# make predictions for test data
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error: %.2f" % mae)
print("R-squared: %.2f" % r2)

joblib.dump(model, "xgboost_model.pkl")