import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau, linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.stats as st

# ===============================
# 1. Load and Subset podatke
# ===============================
# Load  NetCDF file
ds = xr.open_dataset("chirps-v2.0.monthly.nc")

# Subset za Srbiju (CHIRPS latitude descending)
serbia = ds.sel(latitude=slice(42, 46), longitude=slice(19, 23))

# Prosek preko all grid cells
serbia_monthly = serbia.precip.mean(dim=["latitude", "longitude"])

# Convert to DataFrame i sacuvaj
df = serbia_monthly.to_dataframe(name="precip_mm")
df.to_csv("serbia_monthly_rainfall_1981-2025.csv")

# ===============================
# 2. Annual Rainfall Analiza
# ===============================
# Aggregate to annual totals
annual = df.resample("YE").sum()
years = annual.index.year
precip = annual["precip_mm"].values
mean_precip = precip.mean()

# ===============================
# 3. Trend Analysis (Scipy + Scikit-learn)
# ===============================
# Mann-Kendall Trend Test
tau, p_value = kendalltau(years, precip)
print(f"Mann-Kendall p-value: {p_value:.4f}")

# Scipy Linear Regression
slope_scipy, intercept_scipy, _, _, _ = linregress(years, precip)
trend_scipy = intercept_scipy + slope_scipy * years
print(f"Scipy Slope: {slope_scipy:.2f} mm/year")

# Scikit-learn Linear Regression sa Standardization
scaler = StandardScaler()
years_scaled = scaler.fit_transform(np.array(years).reshape(-1, 1))

model = LinearRegression()
model.fit(years_scaled, precip)
trend_sklearn = model.predict(years_scaled)
slope_sklearn = model.coef_[0] / scaler.scale_[0]  # Correct conversion
print(f"Scikit-learn Slope: {slope_sklearn:.2f} mm/year")

# ===============================
# 4. Visualization: Trend Analysis
# ===============================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Scatter plot of observed values
plt.scatter(years, precip, color="blue", alpha=0.7, label="Observed Rainfall")

# Mean precipitation line
plt.axhline(mean_precip, color="green", linestyle=":", linewidth=2, label="1981–2025 Mean")

# Scipy trend line
plt.plot(years, trend_scipy, "r--", label="Trend Line (Scipy)")

# Scikit-learn trend line
plt.plot(years, trend_sklearn, "b--", label="Trend Line (Scikit-learn)")

plt.title("Annual Rainfall in Serbia (1981–2025) with Trends", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Precipitation (mm)", fontsize=12)
plt.xticks(np.arange(1980, 2026, 5))
plt.legend()

# Save and show
plt.savefig("annual_rainfall_trend_sklearn.png", dpi=300, bbox_inches="tight")
plt.show()

# ===============================
# 5. Extreme Event Analysis
# ===============================
drought_threshold = np.percentile(precip, 10)  # 10th percentile
wet_threshold = np.percentile(precip, 90)  # 90th percentile

extreme_dry = annual[annual["precip_mm"] < drought_threshold]
extreme_wet = annual[annual["precip_mm"] > wet_threshold]

print(f"\nDrought Years (≤10th percentile): {len(extreme_dry)}")
print(f"Wet Years (≥90th percentile): {len(extreme_wet)}")

# ===============================
# 6. Extreme Events Visualization
# ===============================
plt.figure(figsize=(12, 6))
plt.bar(years, precip, color="gray", alpha=0.6)
plt.bar(extreme_dry.index.year, extreme_dry["precip_mm"], color="red", label="Drought Years")
plt.bar(extreme_wet.index.year, extreme_wet["precip_mm"], color="blue", label="Wet Years")
plt.axhline(mean_precip, color="black", linestyle="--", linewidth=2, label="Mean")

plt.title("Extreme Rainfall Events in Serbia (1981–2025)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Precipitation (mm)", fontsize=12)
plt.xticks(np.arange(1980, 2026, 5))
plt.legend()

# Save and show
plt.savefig("extreme_events_refined.png", dpi=300, bbox_inches="tight")
plt.show()

# ===============================
# 7. Future Rainfall Prediction with Confidence Interval
# ===============================
future_years = np.arange(2026, 2031).reshape(-1, 1)
future_years_scaled = scaler.transform(future_years)  # Standardize future years
future_predictions = model.predict(future_years_scaled)

# Compute standard error of residuals
y_pred = model.predict(years_scaled)
residuals = precip - y_pred
std_error = np.std(residuals)

# Compute 95% confidence interval
alpha = 0.05
t_value = st.t.ppf(1 - alpha / 2, df=len(years) - 2)  # t-distribution critical value
margin_of_error = t_value * std_error
lower_bound = future_predictions - margin_of_error
upper_bound = future_predictions + margin_of_error

print("Caution: Predictions assume linearity and ignore climate non-stationarity.")
print(f"Predicted Rainfall (2026-2030): {future_predictions}")

# ===============================
# 8. Visualization with Confidence Intervals
# ===============================
plt.figure(figsize=(12, 6))

# Observed rainfall
plt.scatter(years, precip, color="blue", alpha=0.7, label="Observed Rainfall")

# Trend line
plt.plot(years, trend_sklearn, "b--", label="Trend Line (Scikit-learn)")

# Future Predictions
plt.scatter(future_years, future_predictions, color="purple", marker="x", s=100, label="Future Predictions")

# Confidence Interval
plt.fill_between(future_years.ravel(), lower_bound, upper_bound, color="purple", alpha=0.2, label="95% Confidence Interval")

plt.title("Projected Annual Rainfall (2026–2030) with Confidence Intervals", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Precipitation (mm)", fontsize=12)
plt.xticks(np.arange(1980, 2031, 5))
plt.legend()

# Save and show
plt.savefig("future_rainfall_prediction_ci.png", dpi=300, bbox_inches="tight")
plt.show()
