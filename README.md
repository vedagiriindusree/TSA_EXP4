# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 30.09.2025
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('cardekho.csv')

# Aggregate by year (average selling price per year)
X = data.groupby("year")["selling_price"].mean()

# Plot original data
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(X, marker='o')
plt.title('Average Selling Price per Year')
plt.xlabel('Year')
plt.ylabel('Avg Selling Price')
plt.grid()
plt.show()

# Plot ACF and PACF
# Plot ACF and PACF with safe lags
max_lags = min(10, len(X)//4)   # ensures valid range

plt.subplot(2, 1, 1)
plot_acf(X, lags=max_lags, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=max_lags, ax=plt.gca(), method='ywm')
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()


# Fit ARMA(1,1)
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params.get('ar.L1', 0)
theta1_arma11 = arma11_model.params.get('ma.L1', 0)

# Simulate ARMA(1,1) process
N = 1000
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title("ACF of Simulated ARMA(1,1)")
plt.show()

plot_pacf(ARMA_1)
plt.title("PACF of Simulated ARMA(1,1)")
plt.show()

# Fit ARMA(2,2)
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params.get('ar.L1', 0)
phi2_arma22 = arma22_model.params.get('ar.L2', 0)
theta1_arma22 = arma22_model.params.get('ma.L1', 0)
theta2_arma22 = arma22_model.params.get('ma.L2', 0)

# Simulate ARMA(2,2) process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title("ACF of Simulated ARMA(2,2)")
plt.show()

plot_pacf(ARMA_2)
plt.title("PACF of Simulated ARMA(2,2)")
plt.show()
```
### OUTPUT:
# SIMULATED ARMA(1,1) PROCESS:
# Partial Autocorrelation:
<img width="1251" height="666" alt="image" src="https://github.com/user-attachments/assets/d4f49188-19cc-43df-a8b8-6cb1f7db3eea" />

# Autocorrelation:
<img width="1282" height="675" alt="image" src="https://github.com/user-attachments/assets/d8ecabfc-1636-4bf6-b1a9-89a74ca40f33" />


# SIMULATED ARMA(2,2) PROCESS:

# Partial Autocorrelation:
<img width="1252" height="661" alt="image" src="https://github.com/user-attachments/assets/ffab3aa9-5106-42e5-afee-064269a5c335" />

# Autocorrelation:
<img width="1248" height="662" alt="image" src="https://github.com/user-attachments/assets/4dd90aff-0d40-41cc-9f4b-346d8bf84692" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
