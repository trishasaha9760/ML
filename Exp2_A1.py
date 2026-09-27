import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset shape
print("\nShape:")
print(df.shape)

# Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate values
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Target class distribution
print("\nTarget Distribution:")
print(df["target"].value_counts())

# Correlation
print("\nCorrelation Matrix:")
print(df.corr())

# ---------------- VISUALIZATION ----------------

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Wine Class Distribution")
plt.show()

# 2. Histograms
df.hist(figsize=(15, 12), bins=15)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# 3. Boxplot
plt.figure(figsize=(15, 7))
sns.boxplot(data=df.drop("target", axis=1))
plt.xticks(rotation=90)
plt.title("Boxplot of Wine Features")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Pairplot
sns.pairplot(df, hue="target")
plt.show()

# 6. Scatter Plot
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x="alcohol",
    y="flavanoids",
    hue="target",
    data=df
)
plt.title("Alcohol vs Flavanoids")
plt.show()