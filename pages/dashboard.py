import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

st.title("Dashboard Produksi Padi Di Pulau Sumatera")

 # load CSS Style
with open('static/styles.css')as f:
   st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

# Load dataset
padi = pd.read_csv('data/dataset_padi.csv')

with st.expander("HASIL DATA"):
   padi = pd.DataFrame({
      'Provinsi': padi['Provinsi'],
      'Tahun': padi['Tahun'],
      'Produksi': padi['Produksi'],
      'Luas Panen': padi['Luas Panen'],
      'Curah hujan': padi['Curah hujan'],
      'Kelembapan': padi['Kelembapan'],
      'Suhu Rata-rata': padi['Suhu Rata-rata']
   })
   st.dataframe(padi, use_container_width=True)
   st.write(f'<b>NOTES</b>: Sumber data berasal dari Kaggle.', unsafe_allow_html=True)

# Download data csv
download = padi.to_csv(index=False).encode('utf-8')
st.download_button(label='DOWNLOAD DATASET',
                   data = download,
                   key='download_data.csv',
                   file_name='dataset_padi.csv')

# Visualization
with st.expander("VISUALISASI DATA"):
   col1, col2 = st.columns(2)
   with col1:
      padi = pd.DataFrame(padi.groupby(['Provinsi', 'Produksi']).sum().reset_index())
      bar = px.bar(padi, x = 'Provinsi', y = 'Produksi', color = 'Provinsi')
      title = bar.update_layout(title={'text': "Produksi Padi Per Provinsi (Pulau Sumatera)",
                                       'xanchor': 'center',
                                       'yanchor': 'top',
                                       'x': 0.5,
                                       'y': 0.95})
      st.plotly_chart(bar, use_container_width=True)

   with col2:
      padi = pd.DataFrame(padi.groupby(['Tahun', 'Produksi']).sum().reset_index())
      bar = px.bar(padi, x = 'Tahun', y = 'Produksi', color = 'Tahun')
      title = bar.update_layout(title={'text': "Produksi Padi Per Tahun",
                                        'xanchor': 'center',
                                        'yanchor': 'top',
                                        'x': 0.5,
                                        'y': 0.95})
      st.plotly_chart(bar, use_container_width=True)

with st.expander("CORRELATION"):
   col1, col2 = st.columns(2)
   with col1:
      plt.figure(figsize=(15, 8), dpi = 80)
      numeric_cols = padi.select_dtypes(include=[np.number]).columns
      sns.heatmap(padi.loc[:, numeric_cols].corr(), annot=True, cmap='coolwarm', square=True)
      plt.title('Correlation Matrix')
      plt.show()
      st.pyplot(fig=plt)
   
   with col2:
      padi = padi.drop(columns=['Provinsi'])
      corr_df = padi.corr()
      st.dataframe(corr_df, use_container_width=True)
      
   st.write(f'<b>NOTES</b>: Hasil korelasi antar data menunjukkan bahwa data Luas Panen dengan Produksi memiliki nilai korelasi yang positif.', unsafe_allow_html=True)
