import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Calculate correlation matrix
corr = df.corr()

# Generate correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap of Wine Dataset")
plt.tight_layout()
plt.show()

# Find strongest positive correlation
corr_matrix = corr.copy()

# Remove diagonal values
for i in range(len(corr_matrix)):
    corr_matrix.iloc[i, i] = -1

# Find maximum correlation
max_corr = corr_matrix.stack().idxmax()
max_value = corr_matrix.stack().max()

print("Strongest Positive Correlation:")
print("Feature 1:", max_corr[0])
print("Feature 2:", max_corr[1])
print("Correlation:", round(max_value, 4))