# analyzing-data
# 📦 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 🧪 Task 1: Load and Explore the Dataset
try:
    # Load iris dataset from sklearn
    iris_raw = load_iris()
    iris = pd.DataFrame(iris_raw.data, columns=iris_raw.feature_names)
    iris['species'] = iris_raw.target
    iris['species'] = iris['species'].apply(lambda x: iris_raw.target_names[x])

    # Display first few rows
    print("📄 First 5 Rows:")
    print(iris.head())

    # Check structure
    print("\n📊 Data Types and Missing Values:")
    print(iris.info())
    print("\nMissing Values:\n", iris.isnull().sum())

    # Clean dataset (no missing values in this case)
    # iris.dropna(inplace=True)  # Uncomment if using a different dataset with NaNs

except Exception as e:
    print("❌ Error loading dataset:", e)

# 🧮 Task 2: Basic Data Analysis

# Basic statistics
print("\n📈 Basic Statistics:")
print(iris.describe())

# Group by species and compute mean
grouped = iris.groupby('species').mean()
print("\n📊 Mean values grouped by species:")
print(grouped)

# 🧠 Identify patterns
print("\n🔍 Observations:")
print("-> Iris-virginica has the highest average petal length and width.")
print("-> Iris-setosa has the smallest average sepal and petal measurements.")

# 📊 Task 3: Data Visualization

sns.set(style="whitegrid")

# 1. Line Chart (simulate trend with index for demo)
plt.figure(figsize=(10, 5))
plt.plot(iris.index, iris['sepal length (cm)'], label='Sepal Length')
plt.plot(iris.index, iris['petal length (cm)'], label='Petal Length')
plt.title("Line Chart: Sepal and Petal Length over Index")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart: Avg petal length per species
plt.figure(figsize=(7, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(7, 5))
sns.histplot(iris['sepal width (cm)'], kde=True, color='skyblue')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris)
plt.title("Scatter Plot: Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()
