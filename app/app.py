import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("library_data.csv")
print(df.head())

# Check missing values
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Convert Issue_Date to datetime
df['Issue_Date'] = pd.to_datetime(df['Issue_Date'])

top_books = df['Book_Name'].value_counts().head(5)
print("Top 5 Issued Books:")
print(top_books)


popular_category = df['Category'].value_counts()
print("Category-wise Book Issue Count:")
print(popular_category)


df['Month'] = df['Issue_Date'].dt.month
monthly_issues = df['Month'].value_counts().sort_index()
print("Monthly Issue Trend:")
print(monthly_issues)


top_books.plot(kind='bar', color='skyblue')
plt.title("Top 5 Most Issued Books")
plt.xlabel("Book Name")
plt.ylabel("Number of Issues")
plt.show()


popular_category.plot(kind='pie', autopct='%1.1f%%')
plt.title("Category-wise Book Distribution")
plt.ylabel("")
plt.show()


monthly_issues.plot(kind='line', marker='o')
plt.title("Monthly Book Issue Trend")
plt.xlabel("Month")
plt.ylabel("Number of Issues")
plt.show()
