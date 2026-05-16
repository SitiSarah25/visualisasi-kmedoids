import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from sklearn.cluster import KMeans
from kneed import KneeLocator

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Visualisasi Clustering K-Medoids",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stMetric"] {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.stDataFrame {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("Visualisasi Clustering K-Medoids")

st.markdown("""
Sistem visualisasi hasil clustering tingkat kepuasan mahasiswa 
terhadap layanan akademik menggunakan algoritma **K-Medoids**.
""")

# =========================================================
# FILE UPLOAD
# =========================================================
st.markdown("## Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload file Excel hasil clustering (.xlsx)",
    type=["xlsx"]
)

# =========================================================
# MAIN PROGRAM
# =========================================================
if uploaded_file is not None:

    # =====================================================
    # LOAD DATA
    # =====================================================
    df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset berhasil diupload!")

    # =====================================================
    # VALIDASI
    # =====================================================
    if 'cluster' not in df.columns:
        st.error("❌ Dataset harus memiliki kolom 'cluster'")
        st.stop()

    # =====================================================
    # AMBIL KOLOM NUMERIK
    # =====================================================
    indikator_cols = [
        col for col in df.columns
        if col != 'cluster'
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    X = df[indikator_cols]
    y = df['cluster']

    # =====================================================
    # INFORMASI DASAR
    # =====================================================
    jumlah_responden = len(df)
    jumlah_cluster = df['cluster'].nunique()

    # =====================================================
    # METRICS
    # =====================================================
    st.markdown("## Informasi Hasil Clustering")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Responden",
            jumlah_responden
        )

    with col2:
        st.metric(
            "Jumlah Cluster",
            jumlah_cluster
        )

    with col3:
        st.metric(
            "Metode",
            "K-Medoids"
        )

    # =====================================================
    # ELBOW METHOD
    # =====================================================
    st.markdown("## Grafik Elbow Method")

    try:

        K_range = range(1, 11)
        cost = []

        for k in K_range:

            kmedoids = KMedoids(
                n_clusters=k,
                metric='manhattan',
                init='k-medoids++',
                random_state=42
            )

            kmedoids.fit(X)

            cost.append(kmedoids.inertia_)

        # Knee Locator
        kneedle = KneeLocator(
            K_range,
            cost,
            curve='convex',
            direction='decreasing'
        )

        optimal_k = kneedle.knee

        # Plot
        fig_elbow = go.Figure()

        fig_elbow.add_trace(
            go.Scatter(
                x=list(K_range),
                y=cost,
                mode='lines+markers',
                name='Total Cost'
            )
        )

        # Garis optimal K
        fig_elbow.add_vline(
            x=optimal_k,
            line_dash="dash",
            line_color="red"
        )

        fig_elbow.update_layout(
            title=f'Elbow Method (K Optimal = {optimal_k})',
            xaxis_title='Jumlah Cluster (K)',
            yaxis_title='Total Cost / Dissimilarity',
            height=500
        )

        st.plotly_chart(
            fig_elbow,
            use_container_width=True
        )

        st.success(
            f"✅ Jumlah Cluster Optimal Berdasarkan Elbow Method: K = {optimal_k}"
        )

    except:
        st.warning("Grafik Elbow tidak dapat ditampilkan.")

    # =====================================================
    # RATA-RATA CLUSTER
    # =====================================================
    cluster_mean = df.groupby('cluster')[indikator_cols].mean()

    cluster_avg = cluster_mean.mean(axis=1)

    # =====================================================
    # KATEGORI KEPUASAN
    # =====================================================
    kategori = []

    for val in cluster_avg:

        if val >= 4.5:
            kategori.append("Sangat Puas")

        elif val >= 3.5:
            kategori.append("Puas")

        else:
            kategori.append("Cukup Puas")

    summary_df = pd.DataFrame({
        'Cluster': [f'Cluster {i}' for i in cluster_avg.index],
        'Rata-Rata Kepuasan': np.round(cluster_avg.values, 2),
        'Kategori': kategori
    })

    # =====================================================
    # TABEL RINGKASAN
    # =====================================================
    st.markdown("## Ringkasan Cluster")

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    # =====================================================
    # PIE CHART
    # =====================================================
    st.markdown("## Distribusi Responden")

    cluster_counts = df['cluster'].value_counts().sort_index()

    pie_df = pd.DataFrame({
        'Cluster': [f'Cluster {i}' for i in cluster_counts.index],
        'Jumlah': cluster_counts.values
    })

    fig_pie = px.pie(
        pie_df,
        values='Jumlah',
        names='Cluster',
        hole=0.4
    )

    fig_pie.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # =====================================================
    # BAR CHART
    # =====================================================
    st.markdown("## Rata-Rata Kepuasan Tiap Cluster")

    fig_bar = px.bar(
        summary_df,
        x='Cluster',
        y='Rata-Rata Kepuasan',
        text='Rata-Rata Kepuasan'
    )

    fig_bar.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    # =====================================================
    # RADAR CHART
    # =====================================================
    st.markdown("## Radar Chart Per Aspek")

    aspek = {
        'Dosen': indikator_cols[0:4],
        'Tenaga Kependidikan': indikator_cols[4:8],
        'Pengelola Prodi': indikator_cols[8:12],
        'Sarana dan Prasarana': indikator_cols[12:20]
    }

    radar_df = pd.DataFrame()

    for cluster in sorted(df['cluster'].unique()):

        temp = {
            'Cluster': f'Cluster {cluster}'
        }

        for aspek_name, cols in aspek.items():

            temp[aspek_name] = (
                df[df['cluster'] == cluster][cols]
                .mean()
                .mean()
            )

        radar_df = pd.concat(
            [radar_df, pd.DataFrame([temp])],
            ignore_index=True
        )

    categories = list(aspek.keys())

    fig_radar = go.Figure()

    for i in range(len(radar_df)):

        fig_radar.add_trace(
            go.Scatterpolar(
                r=[
                    radar_df.loc[i, 'Dosen'],
                    radar_df.loc[i, 'Tenaga Kependidikan'],
                    radar_df.loc[i, 'Pengelola Prodi'],
                    radar_df.loc[i, 'Sarana dan Prasarana']
                ],
                theta=categories,
                fill='toself',
                name=radar_df.loc[i, 'Cluster']
            )
        )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True,
        height=600
    )

    st.plotly_chart(
        fig_radar,
        use_container_width=True
    )

    # =====================================================
    # EVALUASI CLUSTERING
    # =====================================================
    st.markdown("## Evaluasi Clustering")

    try:

        # ==========================================
        # SILHOUETTE SCORE
        # ==========================================
        sil_score = silhouette_score(
            X,
            y,
            metric='manhattan'
        )

        st.success(
            f"✅ Silhouette Score: {sil_score:.4f}"
        )

        # ==========================================
        # SILHOUETTE VISUALIZATION
        # ==========================================
        sil_values = silhouette_samples(
            X,
            y,
            metric='manhattan'
        )

        fig_sil = go.Figure()

        y_lower = 10

        for i in sorted(y.unique()):

            ith_cluster_sil = sil_values[y == i]

            ith_cluster_sil.sort()

            size_cluster = ith_cluster_sil.shape[0]

            y_upper = y_lower + size_cluster

            fig_sil.add_trace(
                go.Scatter(
                    x=ith_cluster_sil,
                    y=np.arange(y_lower, y_upper),
                    mode='lines',
                    fill='tozerox',
                    name=f'Cluster {i}'
                )
            )

            y_lower = y_upper + 10

        # Garis rata-rata silhouette
        fig_sil.add_vline(
            x=sil_score,
            line_dash="dash",
            line_color="red"
        )

        fig_sil.update_layout(
            title="Silhouette Visualization Hasil Clustering",
            xaxis_title="Silhouette Coefficient",
            yaxis_title="Data",
            height=600
        )

        st.plotly_chart(
            fig_sil,
            use_container_width=True
        )

    except:
        st.warning(
            "Visualisasi silhouette tidak dapat ditampilkan."
        )

    # =====================================================
    # REKOMENDASI MITIGASI
    # =====================================================
    st.markdown("## Rekomendasi Mitigasi")

    # ==========================================
    # HITUNG CLUSTER TERENDAH
    # ==========================================
    rata_rata_indikator = (
        df.groupby('cluster')[indikator_cols]
        .mean()
        .round(2)
    )

    rata_rata_cluster = (
        rata_rata_indikator.mean(axis=1)
        .round(2)
    )

    cluster_terendah = rata_rata_cluster.idxmin()

    st.error(
        f"Cluster dengan tingkat kepuasan terendah adalah "
        f"Cluster {cluster_terendah} "
        f"dengan rata-rata {rata_rata_cluster[cluster_terendah]}"
    )

    # ==========================================
    # REKOMENDASI
    # ==========================================

    with st.expander("1. Sarana Pembelajaran Berbasis Online", expanded=True):

        st.markdown("""
- Melakukan evaluasi dan peningkatan performa LMS agar lebih stabil dan mudah diakses mahasiswa.
- Menyediakan layanan bantuan teknis yang responsif dalam menangani kendala pembelajaran online.
- Mengembangkan fitur pembelajaran yang lebih fleksibel seperti akses ulang materi dan rekaman perkuliahan.
- Melakukan pemeliharaan sistem secara berkala guna menjaga kualitas layanan pembelajaran online.
""")

    with st.expander("2. Peningkatan Responsivitas Tenaga Kependidikan"):

        st.markdown("""
- Meningkatkan kecepatan pelayanan administrasi akademik kepada mahasiswa.
- Menyediakan layanan informasi akademik dengan alur komunikasi yang lebih jelas dan mudah diakses mahasiswa.
""")

    with st.expander("3. Peningkatan Kualitas Pelayanan Pengelola Program Studi"):

        st.markdown("""
- Meningkatkan kejelasan informasi akademik yang diberikan kepada mahasiswa.
- Memastikan pelayanan akademik dilaksanakan secara konsisten sesuai ketentuan yang berlaku.
- Meningkatkan komunikasi akademik antara pengelola program studi dan mahasiswa.
""")

    with st.expander("4. Peningkatan Kepedulian Pengelola Program Studi"):

        st.markdown("""
- Meningkatkan perhatian terhadap kebutuhan dan kendala akademik mahasiswa.
- Menyediakan media komunikasi yang memudahkan mahasiswa dalam menyampaikan konsultasi akademik.
""")

    with st.expander("5. Peningkatan Kepedulian Tenaga Kependidikan"):

        st.markdown("""
- Meningkatkan kualitas interaksi pelayanan kepada mahasiswa secara lebih ramah dan komunikatif.
- Memberikan pendampingan pelayanan administrasi akademik secara lebih responsif.
""")

    # =====================================================
    # DATASET
    # =====================================================
    st.markdown("## Dataset Hasil Clustering")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================
    st.download_button(
        label="⬇️ Download Dataset",
        data=uploaded_file,
        file_name="hasil_clustering.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================================================
# JIKA BELUM UPLOAD
# =========================================================
else:

    st.info(
        "📥 Silakan upload file Excel untuk menampilkan visualisasi clustering."
    )
