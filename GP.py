#%% GP - RTE (Gen 1)
import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from datetime import datetime
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.metrics import mean_squared_error


charging = ["Charging", "Discharging"]


def read_data(file_path, start_time, end_time):
    df = pd.read_csv(file_path, usecols=['Hot circuit 1', 'Cold circuit 1', 'Hot circuit 2', 'Cold circuit 2', 'Flow', 'Time', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'])
    df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
    filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    times = (filtered_df['Time'] - filtered_df['Time'].iloc[0]).dt.total_seconds() / 3600
    Cp = 4.184  # J/g°C
    m = filtered_df['Flow'] / 60
    deltaT = filtered_df['Hot circuit 1'] - filtered_df['Cold circuit 1']

    q = Cp * m * deltaT
    return times, q


def to_datetime(times):
    return datetime.strptime(times, '%Y/%m/%d %H:%M:%S')

def integrate_q_cumtrapz(times, q):
    integral_q = cumtrapz(q, times, initial=0)
    return integral_q

# Actual charging/discharging cycle
data = [
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_1.csv', '2023/09/28 16:17:12', '2023/09/28 17:14:50', '2023/09/29 08:03:18', '2023/09/29 12:27:30'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_2.csv', '2023/10/02 12:05:12', '2023/10/02 13:03:28', '2023/10/02 13:06:48', '2023/10/02 15:05:10'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_3.csv', '2023/10/04 10:51:24', '2023/10/04 14:21:24', '2023/10/05 08:58:40', '2023/10/05 16:55:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_4.csv', '2023/10/09 10:55:52', '2023/10/09 14:14:08', '2023/10/10 09:23:58', '2023/10/10 15:53:28'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_5.csv', '2023/10/19 12:45:30', '2023/10/19 16:01:52', '2023/10/24 12:28:04', '2023/10/24 17:07:02'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_6.csv', '2023/10/25 13:31:00', '2023/10/25 17:00:52', '2023/11/01 12:56:34', '2023/11/01 18:20:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_7.csv', '2023/11/06 14:10:50', '2023/11/06 16:53:52', '2023/11/07 11:47:18', '2023/11/07 18:53:42'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_8.csv', '2023/11/10 10:17:28', '2023/11/10 13:02:34', '2023/11/13 10:29:42', '2023/11/13 15:24:22'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_9.csv', '2023/11/15 09:22:14', '2023/11/15 12:22:16', '2023/11/17 08:22:08', '2023/11/17 12:24:28'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_10.csv', '2023/11/20 10:04:34', '2023/11/20 12:50:52', '2023/11/24 08:48:02', '2023/11/24 12:50:04'),
]


final_times_1 = []  # discharging period
final_qs_1 = []     # total discharging energy
final_times_2 = []  # charging period
final_qs_2 = []     # total charging energy
final_times_3 = []

efficiencies = []  # efficiencies
efficiency_times = []  
final_duration_3 = []

colors = plt.cm.jet(np.linspace(0, 1, len(data)))

for i, entry in enumerate(data):
    # 解析数据项
    test_name = f"Test_{i+1}"
    file_path = entry[0]
    # discharging period
    start1 = to_datetime(entry[3])
    end1 = to_datetime(entry[4])
    # charging period
    start2 = to_datetime(entry[1])
    end2 = to_datetime(entry[2])
    # standing loss period
    start3 = to_datetime(entry[2])
    end3 = to_datetime(entry[3])

    times_1, q_1 = read_data(file_path, start1, end1)
    cumulative_q_1 = integrate_q_cumtrapz(times_1, q_1)
    
    if len(times_1) > 0 and len(cumulative_q_1) > 0:
        final_time_1 = times_1.iloc[-1] 
        final_q_1 = cumulative_q_1[-1] 
        final_times_1.append(final_time_1)
        final_qs_1.append(final_q_1)
    
   
    times_2, q_2 = read_data(file_path, start2, end2)
    cumulative_q_2 = integrate_q_cumtrapz(times_2, q_2)

    duration_3 = (end3 - start3).total_seconds() / 3600  # standing loss period
    final_duration_3.append(duration_3)


    if len(times_2) > 0 and len(cumulative_q_2) > 0:
        final_time_2 = times_2.iloc[-1] 
        final_q_2 = cumulative_q_2[-1] 
        final_times_2.append(final_time_2)
        final_qs_2.append(final_q_2)
    
        efficiency = 100 * final_q_1 / final_q_2
        efficiencies.append(efficiency)

X = np.array(final_duration_3).reshape(-1, 1)
Y = np.array(efficiencies).reshape(-1, 1)

# use RBF kernel，and tune lengthscale parameters
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=20.)  

# create GPR model
model = GPy.models.GPRegression(X,Y,kernel)

# optimize the model
model.optimize(messages=True)

# prediction, increase predict point to get smooth curve
X_pred = np.linspace(0, 180, 2000).reshape(-1, 1)  
Y_pred, Y_var = model.predict(X_pred) # Y_pred is the prediction mean value，Y_var is predict variance

# tests with different color
colors = plt.cm.jet(np.linspace(0, 1, len(X)))

plt.figure(figsize=(10, 6))
plt.plot(X_pred, Y_pred, 'g-', label='GPR Prediction')
plt.fill_between(X_pred.ravel(), Y_pred.ravel() - 1.96 * np.sqrt(Y_var.ravel()), Y_pred.ravel() + 1.96 * np.sqrt(Y_var.ravel()), alpha=0.2, color='orange', label='95% confidence interval')

for i, (x, y) in enumerate(zip(final_duration_3, efficiencies)):
    plt.scatter(x, y, color=colors[i], marker = 'x', s=60, zorder=5, label=f'Test {i+1}')

plt.xlabel('Time (hours)')
plt.ylabel('Round trip efficiency (%)')
plt.xlim(0, 200)
plt.ylim(0, 100)
plt.tick_params(axis = 'both')
#plt.title('Round trip efficiency during Standing Loss Period for Each Test with GPR')
plt.legend(loc='upper right')
plt.savefig('plot_vector.pdf')
plt.show()

#%% GP - RTE (Gen 2)
import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from datetime import datetime
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.metrics import mean_squared_error


charging = ["Charging", "Discharging"]


def read_data(file_path, start_time, end_time):
    df = pd.read_csv(file_path, usecols=['Hot circuit 1', 'Cold circuit 1', 'Flow', 'Time', 'T1'])
    df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
    filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    times = (filtered_df['Time'] - filtered_df['Time'].iloc[0]).dt.total_seconds() / 3600
    Cp = 4.184  # J/g°C
    m = filtered_df['Flow'] / 60
    deltaT = filtered_df['Hot circuit 1'] - filtered_df['Cold circuit 1']

    q = Cp * m * deltaT
    return times, q


def to_datetime(times):
    return datetime.strptime(times, '%Y/%m/%d %H:%M:%S')

def integrate_q_cumtrapz(times, q):
    integral_q = cumtrapz(q, times, initial=0)
    return integral_q

# Actual charging/discharging cycle
data = [
    
    #('/Users/me1mzxx/Desktop/LVV data - Python/Test_1.csv', '2024/04/04 10:25:08', '2024/04/04 12:39:32', '2024/04/04 13:05:22', '2024/04/04 14:14:04'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_1.csv', '2024/05/30 10:46:04', '2024/05/30 12:59:58', '2024/05/30 13:33:48', '2024/05/30 15:29:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_2.csv', '2024/04/09 09:44:56', '2024/04/09 11:56:18', '2024/04/09 16:20:52', '2024/04/09 17:53:48'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_3.csv', '2024/04/30 13:13:56', '2024/04/30 15:31:34', '2024/05/01 10:20:00', '2024/05/01 11:41:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_4.csv', '2024/07/19 10:17:28', '2024/07/19 12:30:26', '2024/07/22 10:22:16', '2024/07/22 11:17:04'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_5.csv', '2024/05/17 10:26:40', '2024/05/17 12:20:46', '2024/05/21 10:00:48', '2024/05/21 10:47:20'),
    #('/Users/me1mzxx/Desktop/LVV data - Python/Test_6.csv', '2024/03/27 10:46:58', '2024/03/27 11:17:24', '2024/04/03 08:42:12', '2024/04/03 12:45:26'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_6.csv', '2024/05/22 10:15:08', '2024/05/22 12:30:26', '2024/05/29 09:44:10', '2024/05/29 10:13:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_7.csv', '2024/07/23 13:02:56', '2024/07/23 15:17:38', '2024/07/25 10:35:28', '2024/07/25 11:47:12')
]


final_times_1 = []  # discharging period
final_qs_1 = []     # total standing loss energy
final_times_2 = []  # charging period
final_qs_2 = []     # total charging energy
final_times_3 = []

efficiencies = []  # efficiencies
efficiency_times = []  
final_duration_3 = []

colors = plt.cm.jet(np.linspace(0, 1, len(data)))

for i, entry in enumerate(data):
    # 解析数据项
    test_name = f"Test_{i+1}"
    file_path = entry[0]
    # discharging period
    start1 = to_datetime(entry[3])
    end1 = to_datetime(entry[4])
    # charging period
    start2 = to_datetime(entry[1])
    end2 = to_datetime(entry[2])
    # standing loss period
    start3 = to_datetime(entry[2])
    end3 = to_datetime(entry[3])

    times_1, q_1 = read_data(file_path, start1, end1)
    cumulative_q_1 = integrate_q_cumtrapz(times_1, q_1)
    
    if len(times_1) > 0 and len(cumulative_q_1) > 0:
        final_time_1 = times_1.iloc[-1] 
        final_q_1 = cumulative_q_1[-1] 
        final_times_1.append(final_time_1)
        final_qs_1.append(final_q_1)
    
   
    times_2, q_2 = read_data(file_path, start2, end2)
    cumulative_q_2 = integrate_q_cumtrapz(times_2, q_2)

    duration_3 = (end3 - start3).total_seconds() / 3600  # standing loss period
    final_duration_3.append(duration_3)


    if len(times_2) > 0 and len(cumulative_q_2) > 0:
        final_time_2 = times_2.iloc[-1] 
        final_q_2 = cumulative_q_2[-1] 
        final_times_2.append(final_time_2)
        final_qs_2.append(final_q_2)
    
        efficiency = 100 * final_q_1 / final_q_2
        efficiencies.append(efficiency)

X = np.array(final_duration_3).reshape(-1, 1)
Y = np.array(efficiencies).reshape(-1, 1)

# use RBF kernel，and tune lengthscale parameters
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=20.)  

# create GPR model
model = GPy.models.GPRegression(X,Y,kernel)

# optimize the model
model.optimize(messages=True)

# prediction, increase predict point to get smooth curve
X_pred = np.linspace(0, 200, 2000).reshape(-1, 1)  
Y_pred, Y_var = model.predict(X_pred) # Y_pred is the prediction mean value，Y_var is predict variance

# tests with different color
colors = plt.cm.jet(np.linspace(0, 1, len(X)))

plt.figure(figsize=(10, 6))
plt.plot(X_pred, Y_pred, 'g-', label='GPR Prediction')
plt.fill_between(X_pred.ravel(), Y_pred.ravel() - 1.96 * np.sqrt(Y_var.ravel()), Y_pred.ravel() + 1.96 * np.sqrt(Y_var.ravel()), alpha=0.2, color='orange', label='95% confidence interval')

for i, (x, y) in enumerate(zip(final_duration_3, efficiencies)):
    plt.scatter(x, y, color=colors[i], marker = 'x', s=60, zorder=5, label=f'Test {i+1}')

plt.xlabel('Time (hours)')
plt.ylabel('Round trip efficiency (%)')
plt.xlim(0, 250)
plt.ylim(0, 100)
plt.tick_params(axis = 'both')
plt.legend(loc='upper right')
plt.savefig('plot_vector.pdf')
plt.show()


# %% GEN1 GP-RTE - actual vs predict and error

import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from datetime import datetime
import GPy

def read_data(file_path, start_time, end_time):
    df = pd.read_csv(file_path, usecols=['Hot circuit 1', 'Cold circuit 1', 'Hot circuit 2', 'Cold circuit 2', 'Flow', 'Time', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'])
    df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
    filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    times = (filtered_df['Time'] - filtered_df['Time'].iloc[0]).dt.total_seconds() / 3600
    Cp = 4.184  # J/g°C
    m = filtered_df['Flow'] / 60
    deltaT = filtered_df['Hot circuit 1'] - filtered_df['Cold circuit 1']

    q = Cp * m * deltaT
    return times, q

def to_datetime(times):
    return datetime.strptime(times, '%Y/%m/%d %H:%M:%S')

def integrate_q_cumtrapz(times, q):
    integral_q = cumtrapz(q, times, initial=0)
    return integral_q

# Actual charging/discharging cycle data
data = [
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_1.csv', '2023/09/28 16:17:12', '2023/09/28 17:14:50', '2023/09/29 08:03:18', '2023/09/29 12:27:30'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_2.csv', '2023/10/02 12:05:12', '2023/10/02 13:03:28', '2023/10/02 13:06:48', '2023/10/02 15:05:10'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_3.csv', '2023/10/04 10:51:24', '2023/10/04 14:21:24', '2023/10/05 08:58:40', '2023/10/05 16:55:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_4.csv', '2023/10/09 10:55:52', '2023/10/09 14:14:08', '2023/10/10 09:23:58', '2023/10/10 15:53:28'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_5.csv', '2023/10/19 12:45:30', '2023/10/19 16:01:52', '2023/10/24 12:28:04', '2023/10/24 17:07:02'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_6.csv', '2023/10/25 13:31:00', '2023/10/25 17:00:52', '2023/11/01 12:56:34', '2023/11/01 18:20:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_7.csv', '2023/11/06 14:10:50', '2023/11/06 16:53:52', '2023/11/07 11:47:18', '2023/11/07 18:53:42'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_8.csv', '2023/11/10 10:17:28', '2023/11/10 13:02:34', '2023/11/13 10:29:42', '2023/11/13 15:24:22'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_9.csv', '2023/11/15 09:22:14', '2023/11/15 12:22:16', '2023/11/17 08:22:08', '2023/11/17 12:24:28'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen 1/Test_10.csv', '2023/11/20 10:04:34', '2023/11/20 12:50:52', '2023/11/24 08:48:02', '2023/11/24 12:50:04'),
]

final_times_1 = []  # discharging period
final_qs_1 = []     # total discharging energy
final_times_2 = []  # charging period
final_qs_2 = []     # total charging energy
final_duration_3 = []

efficiencies = []  # efficiencies
colors = plt.cm.jet(np.linspace(0, 1, len(data)))

for i, entry in enumerate(data):
    # Parsing data entry
    file_path = entry[0]
    start1 = to_datetime(entry[3])
    end1 = to_datetime(entry[4])
    start2 = to_datetime(entry[1])
    end2 = to_datetime(entry[2])
    start3 = to_datetime(entry[2])
    end3 = to_datetime(entry[3])

    times_1, q_1 = read_data(file_path, start1, end1)
    cumulative_q_1 = integrate_q_cumtrapz(times_1, q_1)
    
    if len(times_1) > 0 and len(cumulative_q_1) > 0:
        final_time_1 = times_1.iloc[-1] 
        final_q_1 = cumulative_q_1[-1] 
        final_times_1.append(final_time_1)
        final_qs_1.append(final_q_1)
    
    times_2, q_2 = read_data(file_path, start2, end2)
    cumulative_q_2 = integrate_q_cumtrapz(times_2, q_2)

    duration_3 = (end3 - start3).total_seconds() / 3600
    final_duration_3.append(duration_3)

    if len(times_2) > 0 and len(cumulative_q_2) > 0:
        final_time_2 = times_2.iloc[-1]
        final_q_2 = cumulative_q_2[-1]
        final_times_2.append(final_time_2)
        final_qs_2.append(final_q_2)
    
        efficiency = 100 * final_q_1 / final_q_2
        efficiencies.append(efficiency)

X = np.array(final_duration_3).reshape(-1, 1)
Y = np.array(efficiencies).reshape(-1, 1)

# LOOCV 
predictions = []

for i in range(len(X)):
    # Exclude the ith data point
    X_train = np.delete(X, i, axis=0)
    Y_train = np.delete(Y, i, axis=0)
    
    # Train GPR model
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=20.)
    model = GPy.models.GPRegression(X_train, Y_train, kernel)
    model.optimize(messages=False)
    
    # Predict the excluded point
    X_test = X[i].reshape(1, -1)
    Y_pred, Y_var = model.predict(X_test)
    predictions.append(Y_pred[0][0])

# Estimate recovered energy using predicted RTE
charging_energies = np.array(final_qs_2).reshape(-1, 1)
predicted_rtes = np.array(predictions).reshape(-1, 1)
estimated_recovered_energies = charging_energies * (predicted_rtes / 100)

# Actual recovered energy
actual_recovered_energy = np.array(final_qs_1)

# Calculate differences
differences = np.abs(actual_recovered_energy - estimated_recovered_energies.flatten())

# Plotting
plt.figure(figsize=(10, 6))

# Plot the original data points
for i in range(len(data)):
    plt.scatter(final_duration_3[i], actual_recovered_energy[i], color=colors[i], marker='x', s=40)
    plt.scatter(final_duration_3[i], estimated_recovered_energies[i], color=colors[i], marker='o', s=40)

# Create legend entries
data_length = len(data)
actual_handles = [plt.Line2D([0], [0], marker='x', color=colors[i], linestyle='None') for i in range(data_length)]
predicted_handles = [plt.Line2D([0], [0], marker='o', color=colors[i], linestyle='None') for i in range(data_length)]

# Manually create legend titles
actual_title = plt.Line2D([0], [0], marker='None', color='w', label='Actual', linestyle='None')
predicted_title = plt.Line2D([0], [0], marker='None', color='w', label='Predicted', linestyle='None')

# Add titles to handles list
handles = [actual_title] + actual_handles + [predicted_title] + predicted_handles
labels = ['Actual'] + [f'Test {i+1}' for i in range(data_length)] + ['Predicted'] + [f'Test {i+1}' for i in range(data_length)]

plt.xlabel('Time (hours)')
plt.ylabel('Recovered Energy (kWh)')
plt.xlim(0, 300)
plt.ylim(0, 4.5)
plt.legend(handles, labels, loc='best', ncol=2)
plt.tick_params(axis='both')
plt.savefig('plot_vector.pdf')
plt.show()

# create index
test_indices = list(range(1, len(final_duration_3) + 1))

# 
sorted_data = sorted(zip(test_indices, differences, final_duration_3), key=lambda x: x[2])

# sorted duration
sorted_indices, sorted_errors, sorted_durations = zip(*sorted_data)

# plot error
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_errors)), sorted_errors, color='skyblue', tick_label=[f'Test {idx}' for idx in sorted_indices], width=0.3)
plt.xlabel('Test')
plt.ylabel('Error (kWh)')
plt.xticks(rotation=45)
plt.tick_params(axis='both')
plt.savefig('plot_vector.pdf')
plt.show()


# %% GEN2 GP-RTE - actual vs predict and error

import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from datetime import datetime
import GPy

def read_data(file_path, start_time, end_time):
    df = pd.read_csv(file_path, usecols=['Hot circuit 1', 'Cold circuit 1', 'Flow', 'Time', 'T1'])
    df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
    filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    times = (filtered_df['Time'] - filtered_df['Time'].iloc[0]).dt.total_seconds() / 3600
    Cp = 4.184  # J/g°C
    m = filtered_df['Flow'] / 60
    deltaT = filtered_df['Hot circuit 1'] - filtered_df['Cold circuit 1']

    q = Cp * m * deltaT
    return times, q

def to_datetime(times):
    return datetime.strptime(times, '%Y/%m/%d %H:%M:%S')

def integrate_q_cumtrapz(times, q):
    integral_q = cumtrapz(q, times, initial=0)
    return integral_q

# Actual charging/discharging cycle data
data = [

    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_1.csv', '2024/05/30 10:46:04', '2024/05/30 12:59:58', '2024/05/30 13:33:48', '2024/05/30 15:29:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_2.csv', '2024/04/09 09:44:56', '2024/04/09 11:56:18', '2024/04/09 16:20:52', '2024/04/09 17:53:48'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_3.csv', '2024/04/30 13:13:56', '2024/04/30 15:31:34', '2024/05/01 10:20:00', '2024/05/01 11:41:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_4.csv', '2024/07/19 10:17:28', '2024/07/19 12:30:26', '2024/07/22 10:22:16', '2024/07/22 11:17:04'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_5.csv', '2024/05/17 10:26:40', '2024/05/17 12:20:46', '2024/05/21 10:00:48', '2024/05/21 10:47:20'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_6.csv', '2024/05/22 10:15:08', '2024/05/22 12:30:26', '2024/05/29 09:44:10', '2024/05/29 10:13:32'),
    ('/Users/me1mzxx/Desktop/LVV data - Python/Gen2/Test_7.csv', '2024/07/23 13:02:56', '2024/07/23 15:17:38', '2024/07/25 10:35:28', '2024/07/25 11:47:12')
]

final_times_1 = []  # discharging period
final_qs_1 = []     # total discharging energy
final_times_2 = []  # charging period
final_qs_2 = []     # total charging energy
final_duration_3 = []

efficiencies = []  # efficiencies
colors = plt.cm.jet(np.linspace(0, 1, len(data)))

for i, entry in enumerate(data):
    # Parsing data entry
    file_path = entry[0]
    start1 = to_datetime(entry[3])
    end1 = to_datetime(entry[4])
    start2 = to_datetime(entry[1])
    end2 = to_datetime(entry[2])
    start3 = to_datetime(entry[2])
    end3 = to_datetime(entry[3])

    times_1, q_1 = read_data(file_path, start1, end1)
    cumulative_q_1 = integrate_q_cumtrapz(times_1, q_1)
    
    if len(times_1) > 0 and len(cumulative_q_1) > 0:
        final_time_1 = times_1.iloc[-1] 
        final_q_1 = cumulative_q_1[-1] 
        final_times_1.append(final_time_1)
        final_qs_1.append(final_q_1)
    
    times_2, q_2 = read_data(file_path, start2, end2)
    cumulative_q_2 = integrate_q_cumtrapz(times_2, q_2)

    duration_3 = (end3 - start3).total_seconds() / 3600
    final_duration_3.append(duration_3)

    if len(times_2) > 0 and len(cumulative_q_2) > 0:
        final_time_2 = times_2.iloc[-1]
        final_q_2 = cumulative_q_2[-1]
        final_times_2.append(final_time_2)
        final_qs_2.append(final_q_2)
    
        efficiency = 100 * final_q_1 / final_q_2
        efficiencies.append(efficiency)

X = np.array(final_duration_3).reshape(-1, 1)
Y = np.array(efficiencies).reshape(-1, 1)

# LOOCV
predictions = []

for i in range(len(X)):
    # Exclude the ith data point
    X_train = np.delete(X, i, axis=0)
    Y_train = np.delete(Y, i, axis=0)
    
    # Train GPR model
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=20.)
    model = GPy.models.GPRegression(X_train, Y_train, kernel)
    model.optimize(messages=False)
    
    # Predict the excluded point
    X_test = X[i].reshape(1, -1)
    Y_pred, Y_var = model.predict(X_test)
    predictions.append(Y_pred[0][0])

# Estimate recovered energy using predicted RTE
charging_energies = np.array(final_qs_2).reshape(-1, 1)
predicted_rtes = np.array(predictions).reshape(-1, 1)
estimated_recovered_energies = charging_energies * (predicted_rtes / 100)

# Actual recovered energy
actual_recovered_energy = np.array(final_qs_1)

# Calculate differences
differences = np.abs(actual_recovered_energy - estimated_recovered_energies.flatten())

# Plotting
plt.figure(figsize=(10, 6))

# Plot the original data points
for i in range(len(data)):
    plt.scatter(final_duration_3[i], actual_recovered_energy[i], color=colors[i], marker='x', s=40)
    plt.scatter(final_duration_3[i], estimated_recovered_energies[i], color=colors[i], marker='o', s=40)

# Create legend entries
data_length = len(data)
actual_handles = [plt.Line2D([0], [0], marker='x', color=colors[i], linestyle='None') for i in range(data_length)]
predicted_handles = [plt.Line2D([0], [0], marker='o', color=colors[i], linestyle='None') for i in range(data_length)]

# Manually create legend titles
actual_title = plt.Line2D([0], [0], marker='None', color='w', label='Actual', linestyle='None')
predicted_title = plt.Line2D([0], [0], marker='None', color='w', label='Predicted', linestyle='None')

# Add titles to handles list
handles = [actual_title] + actual_handles + [predicted_title] + predicted_handles
labels = ['Actual'] + [f'Test {i+1}' for i in range(data_length)] + ['Predicted'] + [f'Test {i+1}' for i in range(data_length)]

plt.xlabel('Time (hours)')
plt.ylabel('Recovered Energy (kWh)')
plt.xlim(0, 200)
plt.ylim(0, 4.5)
plt.legend(handles, labels, loc='best', ncol=2)
plt.tick_params(axis='both')
plt.savefig('plot_vector.pdf')
plt.show()

# 创建一个包含测试索引的列表，以便在排序后能保留原始测试编号
test_indices = list(range(1, len(final_duration_3) + 1))

# 将测试编号、错误和持续时间一起打包然后根据持续时间排序
sorted_data = sorted(zip(test_indices, differences, final_duration_3), key=lambda x: x[2])

# 解包排序后的数据
sorted_indices, sorted_errors, sorted_durations = zip(*sorted_data)

# 绘制排序后的错误图
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_errors)), sorted_errors, color='skyblue', tick_label=[f'Test {idx}' for idx in sorted_indices], width=0.2)
plt.xlabel('Test')
plt.ylabel('Error (kWh)')
plt.xticks(rotation=45)
plt.tick_params(axis='both')
plt.savefig('plot_vector.pdf')
plt.show()



