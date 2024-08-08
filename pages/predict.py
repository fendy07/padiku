import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import zscore
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Prediksi Produksi Padi Pulau Sumatera")
st.write("MULTIPLE REGRESSION WITH  SSE, MSE, SSR, MAPE, R2, ADJ[R2], RESIDUAL")

 # load CSS Style
with open('static/styles.css')as f:
   st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

# Load dataset
padi = pd.read_csv('data/dataset_padi.csv')

X = padi[['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu Rata-rata']]
y = padi[['Produksi']]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Preprocessing 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)

# Outlier detection
z_scores = np.abs(zscore(X_train))
threshold = 3
outliers = np.where(z_scores > threshold)

X_train = X_train[(z_scores < threshold).all(axis=1)]
y_train = y_train[(z_scores < threshold).all(axis=1)]

# Modelling with Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Make Predictions
y_pred = linreg.predict(X_test)

# Regression Coefficients
intercept = linreg.intercept_ #Bo
coefficients = linreg.coef_ #B1, B2, B3 dan B4

# Calculate R-Squared Coefficient of determintation
r2 = r2_score(y_test, y_pred)

# Calculate Adjusted R-Squared
n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Calculate SSE, SSR
sse = np.sum((y_test - y_pred) ** 2)
ssr = np.sum((y_pred - np.mean(y)) ** 2)

# Evaluasi MAPE (Mean Average Percentage Error) pada model regresi linear
def calculate_mape(y_test, y_pred):
    y, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Calculate MSE, MAE, RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = calculate_mape(y_test, y_pred)

#regression line
with st.expander("REGRESSION COEFFICIENT"):
   col1, col2, col3, col4, col5 = st.columns(5)
   intercept_value = intercept.item()
   col1.metric('INTERCEPT:', value='{:.4f}'.format(intercept_value), delta = "(Bo)")
   coefficients = coefficients.flatten()
   col2.metric('B1 COEFFICIENT:', value = f'{coefficients[0]:.4f}', delta = " for X1 of Luas Panen (B1)")
   col3.metric('B2 COEFFICIENT', value= f'{coefficients[1]:.4f}', delta = " for X2 of Curah hujan (B2)")
   col4.metric('B3 COEFFICIENT', value = f'{coefficients[2]:.4f}', delta = "for X3 of Kelembapan (B3)")
   col5.metric("B4 COEFFICIENT", value = f'{coefficients[3]:.4f}', delta = "for X4 of Suhu Rata-rata (B4)")
   style_metric_cards(background_color = "#FFFFFF", 
                      border_left_color = "#9900AD", 
                      border_color = "#1f66bd", 
                      box_shadow = "#F71938")
   st.write(f'<b>NOTES</b>: Coefficient menunjukkan pada data fitur sedangkan Intercept adalah hasil interpretasi model yang dilatih.', unsafe_allow_html=True)

# Print R-squared, Adjusted R-squared, and SSE
with st.expander("MEASURE OF VARIATIONS"):
  col1, col2, col3, col4 = st.columns(4)
  col1.metric('R-SQUARED:', value = f'{r2:.4f}', delta = "Coefficient of Determination")
  col2.metric('ADJUSTED R-SQUARED:', value = f'{adjusted_r2:.4f}', delta = "Adj[R2]")
  col3.metric('SUM SQUARED ERROR (SSE):',value= f'{sse:.4f}', delta = "Squared (Actual - Predictions)")
  col4.metric('MAPE:', value = f'{mape:.4f}', delta = "Mean Average Percentage Error")
  
  style_metric_cards(background_color="#FFFFFF", 
                     border_left_color="#9900AD", 
                     border_color="#1f66bd", 
                     box_shadow="#F71938")
  st.write(f'<b>NOTES</b>: Hasil R2 Score menunjukkan hasil yang sangat baik dan pada bagian evaluasi MAPE sudah cukup menurun pada tingkat error yang diterapkan.', unsafe_allow_html=True)

# Print a table with predicted Y
with st.expander("PREDICTION TABLE"):
   result_df = pd.DataFrame({
     'Luas Panen': X_test[:, 0].ravel(),
     'Curah hujan': X_test[:, 1].ravel(),
     'Kelembapan': X_test[:, 2].ravel(),
     'Suhu Rata-rata': X_test[:, 3].ravel(),
     'Produksi | Actual Y': y_test.ravel(),
     'Y_predicted': y_pred.ravel()})
# Check if y_test, y_pred, sse, ssr, mse, mae, rmse have the same length
   if len(y_test) == len(y_pred) == len([sse] * len(y_test)) == len([ssr] * len(y_test)) == len([mse] * len(y_test)) == len([mae] * len(y_test)) == len([rmse] * len(y_test)):
      # Add SSE and SSR to the DataFrame
      result_df['SSE'] = [sse] * len(y_test)
      result_df['SSR'] = [ssr] * len(y_test)
      result_df['MSE'] = [mse] * len(y_test)
      result_df['MAE'] = [mae] * len(y_test)
      result_df['RMSE'] = [rmse] * len(y_test)
   else:
      st.error("Error: All arrays must be of the same length.")

   st.dataframe(result_df, use_container_width=True)
   st.write(f"<b>NOTES</b>: Pada tabel prediksi menggunakan data scaling yang telah diproses sebelumnya sehingga sangat berbeda dengan data aslinya sebelum diolah.", unsafe_allow_html=True)

#download predicted csv
df_download = result_df.to_csv(index = False).encode('utf-8')
st.download_button(label = "DOWNLOAD PREDICTED DATASET", 
                   data = df_download, 
                   key = "download_dataframe.csv", 
                   file_name = "my_dataframe.csv")

with st.expander("RESIDUAL & LINE OF BEST FIT"):
   # Calculate residuals
  residuals = y_test - y_pred
  # Create a new DataFrame to store residuals
  residuals_df = pd.DataFrame({'Actual': y_test.ravel(), 
                               'Predicted': y_pred.ravel(), 
                               'Residuals': residuals.ravel()})
   # Print the residuals DataFrame
  st.dataframe(residuals_df, use_container_width = True)

  col1, col2 = st.columns(2)
  with col1:
   fig = px.scatter(x=y_test[:, 0], y=y_pred[:, 0], labels={'x': 'Actual y | Produksi', 'y': 'Predicted y'})
   # Add a regression line
   z = np.polyfit(y_test[:, 0], y_pred[:, 0], 1)
   p = np.poly1d(z)
   xp = np.linspace(y_test.min(), y_test.max(), 100)
   fig.add_trace(go.Scatter(x=xp, y=p(xp), mode='lines', name='Best Fit Line'))
   fig.update_layout(title = {'text': 'Regression Fit Line', 
                              'xanchor': 'center', 
                              'yanchor': 'top', 
                              'x': 0.5, 
                              'y': 0.95})
   st.plotly_chart(fig, use_container_width=True)
# Residual Plot   
with col2:
   fig = px.scatter(residuals_df, x='Predicted', y='Residuals', trendline='ols', marginal_y='violin')
   fig.update_layout(title = {'text': 'Residual Plot',
                              'xanchor': 'center',
                              'yanchor': 'top',
                              'x': 0.5,
                              'y': 0.95})
   st.plotly_chart(fig, use_container_width=True)

with st.expander('FEATURE IMPORTANCE & NORMALITY TEST'):
   col1, col2 = st.columns(2)
   with col1:
      # Uji Normalitas model Regresi Linear
       y_pred = linreg.predict(X_train)
       err = y_pred - y_train
       hist_data = [err]
       hist_data = [err.ravel() if err.ndim > 1 else err for err in hist_data]
       group_label = ['distplot']
       plot = ff.create_distplot(hist_data, group_label, show_hist=True)
       plot.update_layout(title = {'text': 'Normality Test',
                                   'xanchor': 'center',
                                   'yanchor': 'top',
                                   'x': 0.5, 
                                   'y': 0.95})
       st.plotly_chart(plot, use_container_width=True)
   
with col2:
    # Feature Importance
    importance = permutation_importance(linreg, X_test, y_test, n_repeats=10, random_state=42)
    feature_names = ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu Rata-rata']
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance.importances_mean})
    # Sorting pada dataframe berdasarkan importance
    importance_df = importance_df.sort_values("Importance", ascending=True)
    bar = px.bar(importance_df, x = 'Importance', y = 'Feature')
    title = bar.update_layout(title={'text': "Feature Importance Model Linear Regression", 
                                     'xanchor': 'center', 
                                     'yanchor': 'top', 
                                     'x': 0.5, 
                                     'y': 0.95})
    st.plotly_chart(bar, use_container_width=True)

#----- PREDICT NEW DATA -------
with st.sidebar:
   # Input form new data
   with st.form("input_form", clear_on_submit = True):
      # Luas Panen
      x1 = st.number_input("Luas Panen", ) 
      # Curah hujan
      x2 = st.number_input("Curah hujan", )
      # Kelembapan
      x3 = st.number_input("Kelembapan", )
      # Suhu Rata-rata
      x4 = st.number_input("Suhu Rata-rata", )
      # Submit button for Predict
      submit_button = st.form_submit_button(label = 'Predict')

if submit_button:
   # Make predictions
   new_data = np.array([[x1, x2, x3, x4]])
   new_predict = linreg.predict(new_data)
   # Display prediction
   with st.expander("OPEN"):
      st.write(f"<span style='font-size:34px; color:green;'> Predicted Output: </span> <span style='font-size: 34px;'> {new_predict}</span>", unsafe_allow_html=True)