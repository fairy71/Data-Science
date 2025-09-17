
import pandas as pd

# Sample Sales Data
data = {
    "OrderID": [101, 102, 103, 104, 105],
    "Customer": ["Ali", "Sara", "Ali", "Hina", "Sara"],
    "Product": ["Laptop", "Mobile", "Tablet", "Laptop", "Mobile"],
    "Quantity": [1, 2, 1, 1, 3],
    "Price": [80000, 30000, 20000, 85000, 32000]
}

df = pd.DataFrame(data)


df["Total"] = df["Quantity"] * df["Price"]

print(" Dataset:")
print(df, "\n")


print(" Customer-wise Spending:")
print(df.groupby("Customer")["Total"].sum(), "\n")


print(" Most Sold Product:")
print(df.groupby("Product")["Quantity"].sum().sort_values(ascending=False).head(1), "\n")

print(" Average Order Value:")
print(df["Total"].mean())

