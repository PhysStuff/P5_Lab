import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from symfit import parameters, variables, sin, cos, Fit
import symfit.core.argument
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams.update({'font.size':16})

# Library of potential simple fits

def log(x, a, b, c, d):
    return a*np.log(b*x + c) + d

def line(x, a, b):
    return (x)*a + b

def sincos(x, *param_names):
    n = len(param_names) // 3
    result = 0
    for i in range(n):
        a = param_names[3*i]
        b = param_names[3*i+1]
        c = param_names[3*i+2]
        result += a * np.sin(b * x) + c * np.cos(b * x)
    return result

#Error propogation formula for ratios

def error_calc(a, b , c):
    k = 0.03571428571
    return 2*k*((b**2-a**2)**-0.5 + (c**2-a**2)**-0.5)

# Functions using symfit to fit an arbitrary fourier series to any potential function of N < 1000
# Iterations for series with more than 100 terms not really viable

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def get_params_and_values(model, fit_result):
    param_names = sorted([str(param) for param in model.free_symbols if isinstance(param, symfit.core.argument.Parameter)])
    param_values = [fit_result.params[param] for param in param_names]

    return param_names, param_values

# Experimental data entry

data = pd.read_csv('./Magnet P5 Calibration.csv')

# Adjustment for units read from sensor and power supply

data['Current / mA'] = data['Current / mA']/1000
data['Magnetic Field Strength / mT'] = data['Magnetic Field Strength / mT']*10

data1 = pd.read_csv('./P5 Diffraction Data.csv')

currents = [0, 0.9, 1.7, 2.5, 3.4, 4.4, 5.6, 7.1, 9.7]

# Data Processing to Ratios for Plot of Energy splitting, values averaged across data
# Constants applied for unit scaling

data1['y1'] = (data1['θa1L'])**2 - (data1['θ1L'])**2
data1['x1'] = (data1['θ2L'])**2 - (data1['θ1L'])**2

data1['y2'] = (data1['θ2aL'])**2 - (data1['θ2L'])**2
data1['x2'] = (data1['θ3L'])**2 - (data1['θ2L'])**2

data1['y3'] = (data1['θ1aR'])**2 - (data1['θ1R'])**2
data1['x3'] = (data1['θ2R'])**2 - (data1['θ1R'])**2

data1['y4'] = (data1['θ2aR'])**2 - (data1['θ2R'])**2
data1['x4'] = (data1['θ3R'])**2 - (data1['θ2R'])**2

data1['delta_v'] = (
    data1['y1'] / data1['x1'] +
    data1['y2'] / data1['x2'] +
    data1['y3'] / data1['x3'] +
    data1['y4'] / data1['x4']
    ) / 4 * (
    2.99792458e8 / (2*8.37e-3)
    ) * (6.62607015e-34)

# Using Standard error as precision is the metric being measured

keys = [['θ1L','θa1L','θ2L',],
        ['θ2L','θ2aL','θ3L',],
        ['θ1R','θ1aR','θ2R',],
        ['θ2R','θ2aR','θ3R']]

count = [1, 2, 3, 4]

for n, set in zip(count, keys):
    data1['delta_v_error' + str(n)] = error_calc(data1[set[0]], data1[set[1]], data1[set[2]])

data1['tot_delta_v_error'] = (
    data1['delta_v_error1']**2 +
    data1['delta_v_error2']**2 +
    data1['delta_v_error3']**2 +
    data1['delta_v_error4']**2 )**0.5 * data1['delta_v']

# Uncertainty for Gradient

uncertainty = 0
for n in data1['tot_delta_v_error'][1:]:
    uncertainty += n**2

uncertainty = uncertainty**0.5 / 7

# Define Series and number of terms

x, y = variables('x, y')
w, = parameters('w')
model_dict = {y: fourier_series(x, f=w, n=22)}

# Define data for fit
xdata = data['Current / mA']
ydata = data['Magnetic Field Strength / mT']

# Define a Fit object for this model and data
fit = Fit(model_dict, x=xdata, y=ydata)
fit_result = fit.execute()

# Apply fitted model to the current data from the experiment

mag = fit.model(x=np.array(currents), **fit_result.params).y
data1['Field (mT)'] = mag

mag_error = []

for n in range(0, len(currents)):
    mag_error.append(
        (fit.model(x=np.array([currents[n] + 0.1]), **fit_result.params).y[0] -
         fit.model(x=np.array([currents[n] - 0.1]), **fit_result.params).y[0]) / 2
    )
data1['mag_error'] = mag_error

# Plot Calibration curve and Energy Splitting results

fig1 = plt.figure(figsize=(20,6))
ax1 = fig1.add_subplot(121)
yerr = np.zeros(len(data['Current / mA'])) + 0.1
xerr = yerr

plt.errorbar(
    data['Current / mA'], data['Magnetic Field Strength / mT'],
    xerr=xerr,
    capsize=5,
    ls='none',  color='k',
    marker='o', markersize=5
)
plt.plot(
    np.linspace(0, 10, 2000), fit.model(x=np.linspace(0, 10, 2000), **fit_result.params).y, 
    color='k', ls='--'
)
ax1.set_xbound(0, 10)
ax1.set_ybound(0, 800)

# For use in manual check
#ax1.xaxis.set_minor_locator(AutoMinorLocator(20))

ax1.tick_params(length=7, direction='in', top=True, right=True)

ax1.set_xlabel(r'Supplied Current / A')
ax1.set_ylabel(r'Magnetic Field Strength / mT')

# Fit straight line to values that are linear and uneffected by the low limit inaccuracies
# Account for uncertianties in fit by fitting to minimum and maximum values

part1 = data1.loc[:,['Field (mT)', 'delta_v']]
part2 = data1.loc[:,['Field (mT)', 'delta_v']]
part3 = data1.loc[:,['Field (mT)', 'delta_v']]

for n in range(0, len(part2['delta_v'])):
    part2['delta_v'][n] = part2['delta_v'][n] - data1['tot_delta_v_error'][n]

for n in range(0, len(part3['delta_v'])):
    part3['delta_v'][n] = part3['delta_v'][n] + data1['tot_delta_v_error'][n]

new_columns = ['Field (mT)', 'delta_v']
part1.columns = new_columns
part2.columns = new_columns
part3.columns = new_columns

error_data = pd.concat([part1, part2, part3], ignore_index=True)

error_data.drop(labels=[0,1,9,10,18,19], inplace=True)

popte, pcove = curve_fit(line, error_data['Field (mT)'], error_data['delta_v'] / (1.602176565e-25))
popt, pcov = curve_fit(line, data1['Field (mT)'][2:], data1['delta_v'][2:] / (1.602176565e-25))

fig1 = plt.figure(figsize=(20,6))
ax2 = fig1.add_subplot(122)

plt.errorbar(
    xerr=mag_error[1:], yerr=data1['tot_delta_v_error'][1:] / (1.602176565e-25),
    x=data1['Field (mT)'][1:], y=data1['delta_v'][1:] / (1.602176565e-25),
    capsize=5,
    ls='none', marker='o', color='k'
)
plt.plot(
    np.linspace(data1['Field (mT)'][2] - mag_error[2], data1['Field (mT)'][8] + mag_error[-2], 20), 
    line(np.linspace(data1['Field (mT)'][2] - mag_error[2], data1['Field (mT)'][8] + mag_error[-2], 20), popt[0], popt[1]),
    ls='--', color='k'
)

ax2.set_xbound(0, 850)
ax2.set_ybound(0, 35)

ax2.tick_params(length=7, direction='in', top=True, right=True)

ax2.set_xlabel(r'Magnetic Field Strength / mT')
ax2.set_ylabel(r'Energy Splitting / $\mu$eV')
plt.show()

print(f'The Bohr Magneton yielded is ; {popt[0]*1000} ± {uncertainty / 1.602176565e-25} '+r' $\mu$ eV/T')

display_table = data1.loc[:, ['Voltages (V)', 'Current (A)', 'Field (mT)', 'mag_error', 'delta_v', 'tot_delta_v_error']]

display_table.to_clipboard(index=False)

print(display_table)
