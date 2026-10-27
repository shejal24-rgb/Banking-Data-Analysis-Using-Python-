
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

file_path = Path(__file__).parent / "ProjectDA" / "banking_data.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

required_columns = [
    "Customer_ID", "Age", "Gender", "Account_Type", 
    "Balance", "Transaction_Amount", "City"
]

if not all(col in df.columns for col in required_columns):
    print("Error: Check your CSV column names.")
    print("Available columns:", df.columns.tolist())
    raise SystemExit

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nTotal Customers:", len(df))
print("Average Balance:", np.mean(df["Balance"]))
print("Maximum Balance:", np.max(df["Balance"]))
print("Minimum Balance:", np.min(df["Balance"]))
print("Average Transaction:", np.mean(df["Transaction_Amount"]))

print("\nCustomers by Account Type:")
print(df["Account_Type"].value_counts())

print("\nCustomers by City:")
print(df["City"].value_counts())

print("\nAverage Balance by Account Type:")
print(df.groupby("Account_Type")["Balance"].mean())

sns.countplot(data=df, x="Account_Type")
plt.title("Customers by Account Type")
plt.tight_layout()
plt.show()

sns.barplot(data=df, x="Account_Type", y="Balance")
plt.title("Balance by Account Type")
plt.tight_layout()
plt.show()

sns.barplot(data=df, x="City", y="Balance")
plt.title("Balance by City")
plt.tight_layout()
plt.show()

sns.histplot(data=df, x="Transaction_Amount", bins=10)
plt.title("Transaction Amount Distribution")
plt.tight_layout()
plt.show()

sns.scatterplot(
    data=df,
    x="Balance",
    y="Transaction_Amount",
    hue="Account_Type"
)
plt.title("Balance vs Transaction Amount")
plt.tight_layout()
plt.show()

print("\nBanking Data Analysis Completed!")

