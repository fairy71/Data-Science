import pandas as pd


data = {
    "Student": ["Ali", "Sara", "Hina", "Bilal", "Ayesha"],
    "Math": [78, 92, 67, 85, 90],
    "Science": [88, 79, 72, 95, 85],
    "English": [82, 91, 76, 89, 94]
}

df = pd.DataFrame(data)

print(" Dataset:")
print(df, "\n")


df["Average"] = df[["Math", "Science", "English"]].mean(axis=1)
print(" Student-wise Average Marks:")
print(df[["Student", "Average"]], "\n")


print(" Subject-wise Average:")
print(df[["Math", "Science", "English"]].mean(), "\n")


print(" Top Scorer:")
print(df.loc[df["Average"].idxmax(), ["Student", "Average"]])

