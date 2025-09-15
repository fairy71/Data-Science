Enable desktop notifications for Gmail.
   OK  No thanks
1 of 7
code
Inbox

Ariba Fatima <aribafatima9000@gmail.com>
9:59 PM (2 minutes ago)
to me

import numpy as np
import pandas as pd



np.random.seed(42)  

data = {
    "OrderID": np.arange(1, n+1),
    "CustomerID": np.random.randint(100, 200, n),
    "Product": np.random.choice(["Laptop", "Mobile", "Tablet", "Headphones"], n),
    "Quantity": np.random.randint(1, 5, n),
    "Price": np.random.randint(200, 2000, n),
    "Region": np.random.choice(["North", "South", "East", "West"], n),
    "Date": pd.date_range("2024-01-01", periods=n, freq="D")
}

df = pd.DataFrame(data)

print(" Sample Data:\n", df.head(), "\n")

df.loc[np.random.randint(0, n, 20), "Price"] = np.nan


df["Price"].fillna(df["Price"].mean(), inplace=True)


df.drop_duplicates(inplace=True)


df["TotalSales"] = df["Quantity"] * df["Price"]


df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month


region_sales = df.groupby("Region")["TotalSales"].sum()
print(" Region-wise Sales:\n", region_sales, "\n")


product_sales = df.groupby("Product")["TotalSales"].agg(["sum", "mean"])
print(" Product-wise Sales:\n", product_sales, "\n")


monthly_sales = df.groupby(["Year", "Month"])["TotalSales"].sum()
print(" Monthly Sales:\n", monthly_sales, "\n")


top_customers = df.groupby("CustomerID")["TotalSales"].sum().sort_values(ascending=False).head(5)
print(" Top 5 Customers:\n", top_customers, "\n")

customer_info = pd.DataFrame({
    "CustomerID": np.arange(100, 200),
    "City": np.random.choice(["Lahore", "Karachi", "Islamabad", "Multan"], 100),
    "LoyaltyPoints": np.random.randint(100, 1000, 100)
})

merged_df = pd.merge(df, customer_info, on="CustomerID", how="left")
print(" Merged Data Sample:\n", merged_df.head(), "\n")


pivot = pd.pivot_table(df, values="TotalSales", index="Region", columns="Product", aggfunc="sum")
print("Pivot Table:\n", pivot, "\n")