# Customer Segmentation Using Unsupervised Learning

This project develops a customer segmentation model using unsupervised learning techniques on real-world retail transaction data from the `online_retail_customers` dataset (`clustering.db`)[cite: 1]. By engineering **Recency, Frequency, and Monetary (RFM)** features, customers are grouped via clustering algorithms to identify distinct behavioral profiles and support data-driven marketing strategies[cite: 1].

---

## Project Overview

Understanding customer purchasing behavior is essential for optimizing marketing resources, retaining high-value buyers, and re-engaging inactive accounts[cite: 1]. This project pipeline processes raw transaction records, constructs RFM features, standardizes the metrics, and applies K-Means clustering to segment customers effectively.

### Expected Business Deliverables
* **Champions & Loyalists:** Identification of high-value customers for targeted retention and VIP perks[cite: 1].
* **At-Risk / Inactive Customers:** Detection of churn risks requiring targeted win-back campaigns[cite: 1].
* **Low-Value / One-Time Buyers:** Recognition of single-purchase shoppers for tailored communication[cite: 1].
* **Strategic Resource Allocation:** Data-driven insights to optimize marketing budgets across distinct customer tiers[cite: 1].

---

## Data Dictionary (`online_retail_customers`)

* **InvoiceNo:** Unique identifier for each transaction (Invoices starting with 'C' indicate cancellations)[cite: 1].
* **StockCode:** Unique product or item code[cite: 1].
* **Description:** Text description of the product[cite: 1].
* **Quantity:** Number of units purchased in the transaction[cite: 1].
* **InvoiceDate:** Date and time when the invoice was generated[cite: 1].
* **UnitPrice:** Price per unit of the product in GBP (British Pounds)[cite: 1].
* **CustomerID:** Unique identifier for each customer[cite: 1].
* **Country:** Country where the customer resides[cite: 1].

---

## Instructions for Data Cleaning & Feature Engineering

1. **Filter Missing IDs:** Remove all rows where `CustomerID` is missing[cite: 1].
2. **Datetime Conversion:** Ensure `InvoiceDate` is properly parsed as a datetime object[cite: 1].
3. **Exclude Cancellations:** Remove transactions where `InvoiceNo` starts with `'C'`[cite: 1].
4. **Create TotalPrice:** Calculate total transaction value via $\text{Quantity} \times \text{UnitPrice}$[cite: 1].
5. **Reference Date Setup:** Choose a reference date one day after the most recent invoice date to calculate recency[cite: 1].
6. **RFM Aggregation:** Group data by `CustomerID` to compute[cite: 1]:
   * **Recency:** Days since the last purchase[cite: 1].
   * **Frequency:** Number of unique invoices per customer[cite: 1].
   * **Monetary:** Total spending by each customer (sum of `TotalPrice`)[cite: 1].

---

## Python Implementation

```python
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load Data from Database
conn = sqlite3.connect('clustering.db')
df = pd.read_sql_query("SELECT * FROM online_retail_customers;", conn)
conn.close()

# 2. Data Cleaning
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 3. RFM Feature Engineering
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# 4. Feature Scaling & Clustering
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
