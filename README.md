import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv('/Users/zhanglinxuan/Desktop/Dataset (FV)/Dataset (RP).csv')




column_data = data['NO2_emission']
stats.probplot(column_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of ' + 'NO2_emission')
plt.show()

column_data = data['PM10_emission']
stats.probplot(column_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of ' + 'PM10 emission')
plt.show()

column_data = data['NO_emission']
stats.probplot(column_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of ' + 'NO_emission')
plt.show()

data['log_NO2_emission'] = np.log(data['NO2_emission'])
column_data = data['log_NO2_emission']
stats.probplot(column_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of ' + 'log_NO2_emission')
plt.show()

data['log_PM10_emission'] = np.log(data['PM10_emission'])
column_data = data['log_PM10_emission']
stats.probplot(column_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of ' + 'log_PM10_emission')
plt.show()

data['log_NO_emission'] = np.log(data['NO_emission'])
column_data = data['log_NO_emission']
stats.probplot(column_data, dist="norm", plot=plt)
plt.title('Q-Q Plot of ' + 'log_NO_emission')
plt.show()



model1_log = smf.mixedlm("log_NO2_emission ~ NDVI + Park_size + Distance_to_Park + Temperature + Wind_speed+ NDVI_Temp_Interaction + NDVI_Wind_speed_Interaction", 
                    data, 
                    groups=data["Monitoring_site"]) 

result1_log = model1_log.fit()


print(result1_log.summary())

var_resid = result1_log.scale
var_random_effect = float(result1_log.cov_re.iloc[0, 0])
var_fixed_effect = result1_log.predict(data).var()

total_var = var_fixed_effect + var_random_effect + var_resid

conditional_r1 = (var_fixed_effect + var_random_effect) / total_var


print(f"Conditional R^2: {conditional_r1}")





data['log_PM10_emission'] = np.log(data['PM10_emission'])
data['NDVI_Temp_Interaction'] = data['NDVI'] * data['Temperature']
data['NDVI_Wind_speed_Interaction'] = data['NDVI'] * data['Wind_speed']

model2_log = smf.mixedlm("log_PM10_emission ~ NDVI + Park_size + Distance_to_Park + Temperature + Wind_speed+ NDVI_Temp_Interaction + NDVI_Wind_speed_Interaction", 
                    data, 
                    groups=data["Monitoring_site"]) 

result2_log = model2_log.fit()


print(result2_log.summary())


ar_resid = result2_log.scale
var_random_effect = float(result2_log.cov_re.iloc[0, 0])
var_fixed_effect = result2_log.predict(data).var()


total_var = var_fixed_effect + var_random_effect + var_resid

marginal_r2 = var_fixed_effect / total_var
conditional_r2 = (var_fixed_effect + var_random_effect) / total_var


print(f"Conditional R^2: {conditional_r2}")





data['log_NO_emission'] = np.log(data['NO_emission'])
data['NDVI_Temp_Interaction'] = data['NDVI'] * data['Temperature']
data['NDVI_Wind_speed_Interaction'] = data['NDVI'] * data['Wind_speed']

model3_log = smf.mixedlm("log_NO_emission ~ NDVI + Park_size + Distance_to_Park + Temperature + Wind_speed+ NDVI_Temp_Interaction + NDVI_Wind_speed_Interaction", 
                    data, 
                    groups=data["Monitoring_site"]) 

result3_log = model3_log.fit()


print(result3_log.summary())


var_resid = result3_log.scale
var_random_effect = float(result3_log.cov_re.iloc[0, 0])
var_fixed_effect = result3_log.predict(data).var()

total_var = var_fixed_effect + var_random_effect + var_resid

marginal_r3 = var_fixed_effect / total_var
conditional_r3 = (var_fixed_effect + var_random_effect) / total_var


print(f"Conditional R^2: {conditional_r3}")






