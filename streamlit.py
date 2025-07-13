import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score

# --- Konfigurasi Tampilan Kecil ---
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7
})

# --- Fungsi Top 5 State ---
def get_top5_states(df, cluster_name):
    df_filtered = df[df['cluster_name'] == cluster_name]
    return (
        df_filtered['seller_state']
        .value_counts()
        .head(5)
        .reset_index(name='count')
        .rename(columns={'index': 'seller_state'})
    )

# --- Setup Halaman ---
st.set_page_config(page_title="Olist Clustering Dashboard", layout="wide")
st.title("📊 Olist Seller Clustering Dashboard")
st.markdown("Dashboard ini menampilkan visualisasi data seller Olist berdasarkan jumlah cluster")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("df_seller.csv")
    df['first_order_month'] = pd.to_datetime(df['first_order_month'], errors='coerce')
    df['last_order_month'] = pd.to_datetime(df['last_order_month'], errors='coerce')
    return df

df_seller = load_data()

# --- Sidebar Input ---
st.sidebar.header("Pilih Jumlah Cluster dan Visualisasi")
n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)
plot_option = st.sidebar.selectbox(
    "Jenis Visualisasi",
    [
        "Barplot median per Cluster",
        "Distribusi Fitur",
        "Distribusi Cluster",
        "Total Revenue per Cluster",
        "Distribusi Active Months per Cluster",
        "Top 5 State per Cluster"
    ]
)

# --- Hitung LRFM ---
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

# --- Clustering ---
st.subheader("Clustering KMeans")

df_cluster = df_lrfm.copy()
df_cluster['review'] = df_seller['average_review_score']
df_cluster['time_process'] = df_seller['processing_time_days']

X = df_cluster[['length', 'recency', 'frequency', 'monetary', 'review', 'time_process']].copy()
X['monetary_log'] = np.log1p(X['monetary'])

X_robust = X[['length', 'recency', 'frequency', 'review', 'time_process']]
X_standard = X[['monetary_log']]
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()
X_robust_scaled = robust_scaler.fit_transform(X_robust)
X_standard_scaled = standard_scaler.fit_transform(X_standard)

X_scaled = pd.DataFrame(
    np.hstack([X_robust_scaled, X_standard_scaled]),
    columns=list(X_robust.columns) + ['monetary'],
    index=X.index
)

model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
clusters = model.fit_predict(X_scaled)
df_cluster['cluster'] = clusters
df_seller['cluster'] = clusters  # sinkronkan dengan df_seller

# --- Silhouette dan Elbow ---
sil_score = silhouette_score(X_scaled, clusters)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ℹ Silhouette Score")
    st.success(f"Silhouette Score untuk {n_clusters} cluster adalah: *{sil_score:.3f}*")

with col2:
    st.markdown("### 📉 Elbow Graph")
    inertias = []
    cluster_range = range(2, 11)
    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    fig_elbow, ax = plt.subplots(figsize=(4, 3))
    ax.plot(cluster_range, inertias, marker='o', linestyle='-', color='teal')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Curve')
    ax.grid(True)
    st.pyplot(fig_elbow)

# --- Visualisasi Cluster ---
st.subheader("Visualisasi Cluster")

cluster_name_map = {0: 'Regular Seller', 1: 'Potensial Seller', 2: 'High-Value Seller'}
df_cluster['cluster_name'] = df_cluster['cluster'].map(cluster_name_map)
df_seller['cluster_name'] = df_cluster['cluster_name']
df_seller['cluster'] = df_cluster['cluster']

if plot_option == "Barplot median per Cluster":
    st.markdown("### Pilih Fitur untuk Barplot")
    selected_feature = st.selectbox(
        "Fitur yang Ditampilkan:",
        ["length", "recency", "frequency", "monetary"]
    )

    median_per_cluster = df_cluster.groupby('cluster')[selected_feature].median().reset_index()
    fig_bar, ax = plt.subplots(figsize=(4, 2.5))
    sns.barplot(data=median_per_cluster, x='cluster', y=selected_feature, palette='pastel', ax=ax)
    ax.set_title(f'Median {selected_feature.capitalize()} per Cluster')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=7)
    st.pyplot(fig_bar)

elif plot_option == "Distribusi Fitur":
    median_matrix = df_cluster.groupby('cluster')[['length', 'recency', 'frequency', 'monetary','review','time_process']].median()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(median_matrix.T, annot=True, cmap="Blues", fmt=".1f", ax=ax)
    st.pyplot(fig)

elif plot_option == "Distribusi Cluster":
    fig, ax = plt.subplots(figsize=(3,3))
    df_cluster['cluster'].value_counts().plot.pie(
        autopct='%1.1f%%', ax=ax, ylabel='', title='Distribusi Seller per Cluster'
    )
    st.pyplot(fig)

elif plot_option == "Total Revenue per Cluster":
    revenue_per_cluster = df_seller.groupby('cluster_name')['monetary'].sum().reset_index()
    fig_rev, ax = plt.subplots(figsize=(4, 2.5))
    sns.barplot(data=revenue_per_cluster, x='cluster_name', y='monetary', palette='viridis', ax=ax)
    ax.set_title('Total Revenue per Cluster')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:,.0f}', (p.get_x() + p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=7)
    st.pyplot(fig_rev)

elif plot_option == "Distribusi Active Months per Cluster":
    fig_box, ax2 = plt.subplots(figsize=(4, 2.5))
    sns.boxplot(data=df_seller, x='cluster_name', y='active_months', palette='Set2', ax=ax2)
    ax2.set_title('Distribusi Active Months per Cluster')
    st.pyplot(fig_box)

elif plot_option == "Top 5 State per Cluster":
    st.markdown(f"### Top 5 State Terbanyak Tiap Cluster (Total: {n_clusters} Cluster)")

    # Loop semua cluster
    num_cols = 3  # banyak kolom per baris (bisa diubah ke 2 jika ingin lebih besar)
    rows = (n_clusters + num_cols - 1) // num_cols  # jumlah baris
    
    fig, axs = plt.subplots(rows, num_cols, figsize=(4 * num_cols, 2.5 * rows))
    axs = axs.flatten()

    for c in range(n_clusters):
        cluster_label = f"Cluster {c}"
        df_filtered = df_seller[df_seller['cluster'] == c]
        top5_states = (
            df_filtered['seller_state']
            .value_counts()
            .head(5)
            .reset_index(name='count')
            .rename(columns={'index': 'seller_state'})
        )

        sns.barplot(data=top5_states, x='count', y='seller_state', ax=axs[c], palette='pastel')
        axs[c].set_title(f"{cluster_label}", fontsize=9)
        axs[c].set_xlabel("Jumlah Seller")
        axs[c].set_ylabel("State")

        for p in axs[c].patches:
            width = p.get_width()
            axs[c].annotate(f'{int(width)}', (p.get_x() + width + 0.3, p.get_y() + p.get_height() / 2),
                            ha='left', va='center', fontsize=7)

    # Kosongkan sisa subplot jika jumlah cluster tidak pas dengan grid
    for i in range(n_clusters, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


# --- Output Dataframe ---
st.subheader("📄 Data Seller dengan Cluster")
st.dataframe(df_cluster.head(10))