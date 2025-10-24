# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
df = sns.load_dataset("titanic")

# Display first few rows
print("First 5 rows of data:")
print(df.head())

# --- Basic Data Info ---
print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# --- Data Cleaning ---
# Fill missing ages with median
df['age'].fillna(df['age'].median(), inplace=True)

# Drop rows where 'embarked' is missing
df.dropna(subset=['embarked'], inplace=True)

# --- Feature Engineering ---
# Create a new column: 'family_size'
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Use NumPy for a new column based on age
df['age_group'] = np.where(df['age'] < 18, 'Child', 
                   np.where(df['age'] < 60, 'Adult', 'Senior'))

# --- Basic Analysis ---
print("\nSurvival rate by gender:")
print(df.groupby('sex')['survived'].mean())

print("\nAverage age by class:")
print(df.groupby('class')['age'].mean())

# --- Visualization ---

# 1. Survival count by gender
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

# 2. Age distribution by class
sns.boxplot(x='class', y='age', data=df)
plt.title("Age Distribution by Passenger Class")
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 4. Family size vs. survival
sns.barplot(x='family_size', y='survived', data=df)
plt.title("Family Size vs Survival Rate")
plt.show()
