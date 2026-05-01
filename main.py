import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load real dataset
df = pd.read_csv("data/churn_data.csv")

print(df.head())

# -------------------------
# EDA (Exploratory Data Analysis)
# -------------------------

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Distribution of churn
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.savefig("outputs/churn_distribution.png")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("outputs/monthly_charges_vs_churn.png")
plt.show()

# Support Calls vs Churn
sns.boxplot(x='Churn', y='SupportCalls', data=df)
plt.title("Support Calls vs Churn")
plt.savefig("outputs/support_calls_vs_churn.png")
plt.show()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# Save model
joblib.dump(model, "models/churn_model.pkl")

print("\nModel saved!")

# Feature Importance
importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feat_df)

sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.show()

# -------------------------
# SAFE DATA CLEANING
# -------------------------

# Check available columns
print("\nColumns in dataset:", df.columns)

# Handle TotalCharges only if it exists
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop customerID if exists
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Drop missing values
df.dropna(inplace=True)