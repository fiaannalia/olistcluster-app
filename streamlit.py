import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score

# --- Setup Halaman ---
st.set_page_config(page_title="LRFM Clustering Dashboard", layout="wide")

st.title("📊 LRFM Clustering Dashboard")
st.markdown("Aplikasi ini menghitung nilai *Length, Recency, Frequency, Monetary (LRFM)* lalu menerapkan *KMeans Clustering* terhadap seller Olist.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("df_seller.csv")
    df['first_order_month'] = pd.to_datetime(df['first_order_month'], errors='coerce')
    df['last_order_month'] = pd.to_datetime(df['last_order_month'], errors='coerce')
    return df

df_seller = load_data()

# --- Sidebar Input ---
st.sidebar.header("🔧 Pengaturan")
n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)
plot_option = st.sidebar.selectbox("Pilih Jenis Visualisasi", ["Barplot Mean per Cluster", "Distribusi Fitur (Heatmap)", "Komposisi Cluster (Pie Chart)"])

# --- Hitung LRFM ---
st.subheader("Step 1: Hitung Nilai LRFM")

analysis_date = df_seller['last_order_month'].max() + pd.DateOffset(months=1)

df_seller['length'] = (
    (df_seller['last_order_month'].dt.year - df_seller['first_order_month'].dt.year) * 12 +
    (df_seller['last_order_month'].dt.month - df_seller['first_order_month'].dt.month)
) + 1

df_seller['recency'] = (
    (analysis_date.year - df_seller['last_order_month'].dt.year) * 12 +
    (analysis_date.month - df_seller['last_order_month'].dt.month)
)

df_seller['frequency'] = df_seller['total_orders']
df_seller['monetary'] = df_seller['total_payment_value']

df_lrfm = df_seller[['seller_id', 'length', 'recency', 'frequency', 'monetary']]
st.dataframe(df_lrfm.head())

# --- Clustering ---
st.subheader("Step 2: Clustering KMeans")

df_cluster = df_lrfm.copy()
df_cluster['review'] = df_seller['average_review_score']
df_cluster['time_process'] = df_seller['processing_time_days']

# --- Feature Matrix ---
X = df_cluster[['length', 'recency', 'frequency', 'monetary', 'review', 'time_process']].copy()

# Transformasi log untuk monetary
X['monetary_log'] = np.log1p(X['monetary'])

# Pisahkan kolom untuk scaling
X_robust = X[['length', 'recency', 'frequency', 'review', 'time_process']]
X_standard = X[['monetary_log']]

# Scaling
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()

X_robust_scaled = robust_scaler.fit_transform(X_robust)
X_standard_scaled = standard_scaler.fit_transform(X_standard)

# Gabungkan kembali
X_scaled = pd.DataFrame(
    data=np.hstack([X_robust_scaled, X_standard_scaled]),
    columns=list(X_robust.columns) + ['monetary'],
    index=X.index
)

# KMeans
model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
clusters = model.fit_predict(X_scaled)
df_cluster['cluster'] = clusters

# Silhouette Score
sil_score = silhouette_score(X_scaled, clusters)
st.success(f"Silhouette Score: *{sil_score:.3f}*")

# --- Visualisasi ---
st.subheader("Step 3: Visualisasi Cluster")

if plot_option == "Barplot Mean per Cluster":
    mean_per_cluster = df_cluster.groupby('cluster')[['length', 'recency', 'frequency', 'monetary']].mean()
    st.bar_chart(mean_per_cluster)

elif plot_option == "Distribusi Fitur (Heatmap)":
    mean_matrix = df_cluster.groupby('cluster')[['length', 'recency', 'frequency', 'monetary']].mean()
    fig, ax = plt.subplots()
    sns.heatmap(mean_matrix.T, annot=True, cmap="Blues", fmt=".1f", ax=ax)
    st.pyplot(fig)

elif plot_option == "Komposisi Cluster (Pie Chart)":
    fig, ax = plt.subplots()
    df_cluster['cluster'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, ylabel='', title='Komposisi Seller per Cluster')
    st.pyplot(fig)

# --- Preview Dataframe Output ---
st.subheader("📄 Data Seller dengan Cluster")
st.dataframe(df_cluster.head(10))
