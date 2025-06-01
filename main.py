import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Import and preprocess the data
df = pd.read_csv('Housing.csv')

print(df.head())    #to test if loaded correctly

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

#preprocess, yes/no into 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

#Split data into train test sets
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# dfine features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Show dataset shapes
print("\nTrain/test split shapes:")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# Step 6: Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#Fit a linear regression model
y_pred = lr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

#Evaluate mode
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("R² Score:", round(r2, 4))

#Plot regression line and interpret coefficients
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="skyblue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nModel Coefficients:\n")
print(coefficients)
