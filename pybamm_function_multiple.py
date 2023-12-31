import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as tm 
import random

"""

This function takes in an input array containing Fourier constants for a current 
waveform and outputs the terminal voltage for a battery (along with the corresponding times).

The "inputs" array is a n*15 array, with each row corresponding to a particular input current 
waveform. Each row has 15 elements: the first 7 (elements 0-6) are the amplitudes of each of the sine waves
that make up the input waveform, the following 7 (elements 7-13) are the corresponding frequencies, and 
the last element is a constant. The input waveform is essentially a Fourier series truncated at 7 sine 
terms.
"""
def pybamm_SPMe_Sim(inputs):
    model_SPMe_1 = pybamm.lithium_ion.SPMe()
    var_pts_1 = {
        "x_n": 50,  # negative electrode
        "x_s": 25,  # separator 
        "x_p": 50,  # positive electrode
        "r_n": 50,  # negative particle
        "r_p": 50,  # positive particle
    }
    parameters_1 = model_SPMe_1.default_parameter_values
    

    def my_fun(A_values, B_values, omega_values, constant):
        def current(t):
            sine_terms = np.sum([A * pybamm.sin(2 * np.pi * omega * t) for A, omega in zip(A_values, omega_values)], axis=0)
            cosine_terms = np.sum([B * pybamm.cos(2 * np.pi * omega * t) for B, omega in zip(B_values, omega_values)], axis=0)
            return sine_terms + cosine_terms + constant

        return current

    num_inputs = inputs.shape[0]

    terminal_voltage_list = []
    currentData_list = []
    time_vals_list = []

    start_time = tm.time()

    for i in range(num_inputs):
        print("running for input #"+str(i)+" of "+str(num_inputs))
        A_values = inputs[i, :7]
        B_values = inputs[i, 7:14]
        omega_values = inputs[i, 14:21]
        constant = inputs[i, 21]

        parameters_1['Current function [A]'] = my_fun(A_values, B_values, omega_values, constant)

        thetas = inputs[i, 21:]
        
        parameters_1['Cation transference number'] = 0.38
        parameters_1['Cell volume [m3]'] = 3.914e-5
        parameters_1['Electrode height [m]'] = 0.4527
        parameters_1['Electrode width [m]'] = 0.4527
        parameters_1['Electrolyte conductivity [S.m-1]'] = thetas[0]#1.3 #****
        parameters_1['Electrolyte diffusivity [m2.s-1]'] = thetas[1]#5.35e-10 #****
        parameters_1['Initial concentration in negative electrode [mol.m-3]'] = 2392.8
        parameters_1['Initial concentration in positive electrode [mol.m-3]'] = 28595
        parameters_1['Maximum concentration in negative electrode [mol.m-3]'] = 28746
        parameters_1['Maximum concentration in positive electrode [mol.m-3]'] = 35380
        parameters_1['Negative electrode diffusivity [m2.s-1]'] = thetas[2]
        parameters_1['Negative electrode thickness [m]'] = 6.2e-5
        parameters_1['Negative particle radius [m]'] = 2.5e-6
        parameters_1['Nominal cell capacity [A.h]'] = 4.9872
        parameters_1['Positive electrode diffusivity [m2.s-1]'] = thetas[3]
        parameters_1['Positive electrode thickness [m]'] = 6.7e-5
        parameters_1['Positive particle radius [m]'] = 3.5e-6
        parameters_1['Separator porosity'] = 0.4
        parameters_1['Separator thickness [m]'] = 1.2e-5
        parameters_1['Negative electrode active material volume fraction'] = 1
        parameters_1['Positive electrode active material volume fraction'] = 1

        sim_1 = pybamm.Simulation(model_SPMe_1, parameter_values=parameters_1, var_pts=var_pts_1)
        sim_1.solve([0, 600])

        terminal_voltage = sim_1.solution['Terminal voltage [V]'].data
        currentData = sim_1.solution['Current [A]'].data
        time_vals = sim_1.solution['Time [s]'].data

        terminal_voltage_list.append(terminal_voltage)
        currentData_list.append(currentData)
        time_vals_list.append(time_vals)

    end_time = tm.time()
    execution_time = end_time - start_time
    print(f"Execution time for {num_inputs} input tuples: {execution_time:.2f} seconds")

    return terminal_voltage_list, currentData_list, time_vals_list


"""
The input array is a n*15 array, with each row corresponding to a 
particular input current waveform. Each row has 15 elements: the first 
7 (elements 0-6) are the amplitudes of each of the sine waves that 
make up the input waveform, the following 7 (elements 7-13) are 
the corresponding frequencies, and the last element is a constant. The 
input waveform is essentially a Fourier series truncated at 7 sine terms.
"""
# batCap = 4.9872

# inputs = np.array([
#     [-batCap/2, batCap/3, -batCap/4, batCap/5, -batCap/6, batCap/7, -batCap/8, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/2],
#     [batCap/3, -batCap/4, batCap/5, -batCap/6, batCap/7, -batCap/8, batCap/9, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/3],
#     [-batCap/4, batCap/5, -batCap/6, batCap/7, -batCap/8, batCap/9, -batCap/10, 0.55, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/4],
#     [batCap/5, -batCap/6, batCap/7, -batCap/8, batCap/9, -batCap/10, batCap/11, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/5],
#     [-batCap/6, batCap/7, -batCap/8, batCap/9, -batCap/10, batCap/11, -batCap/12, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/6]
# ])


# #calling the function and storing the outputs (as well as the current in its full time form rather than just as Fourier constants)
# terminal_voltage_list, currentData_list, time_vals_list = pybamm_SPMe_Sim(inputs)


# print("Size of input and output lists:", len(terminal_voltage_list))


# # Plotting
# plt.rcParams['figure.dpi'] = 800
# for i in range(len(time_vals_list)):
#     terminal_voltage = terminal_voltage_list[i]
#     currentData = currentData_list[i]
#     time_vals = time_vals_list[i]

#     fig, ax1 = plt.subplots()
#     ax1.plot(time_vals, terminal_voltage, label='Terminal Voltage', color='blue')
#     ax1.set_xlabel('Time [s]')
#     ax1.set_ylabel('Terminal Voltage [V]', color='blue')
#     ax1.tick_params('y', colors='blue')
    
#     ax2 = ax1.twinx()
#     ax2.plot(time_vals, currentData, label='Current', color='orange')
#     ax2.set_ylabel('Current [A]', color='orange')
#     ax2.tick_params('y', colors='orange')
    
#     plt.title(f'Terminal Voltage and Current for {i+1}th element in input list')
#     plt.legend()
#     plt.show()
batCap = 4.9872 
inputs = np.array([
    [-batCap/2, batCap/3, -batCap/4, batCap/5, -batCap/6, batCap/7, -batCap/8, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/2],
    [batCap/3, -batCap/4, batCap/5, -batCap/6, batCap/7, -batCap/8, batCap/9, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/3],
    [-batCap/4, batCap/5, -batCap/6, batCap/7, -batCap/8, batCap/9, -batCap/10, 0.55, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/4],
    [batCap/5, -batCap/6, batCap/7, -batCap/8, batCap/9, -batCap/10, batCap/11, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/5],
    [-batCap/6, batCap/7, -batCap/8, batCap/9, -batCap/10, batCap/11, -batCap/12, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/6]
])
#%% setting up OED problem
# params (theta) are in order: [el_cond, el_diff]
#         parameters_1['Electrolyte conductivity [S.m-1]'] = 1.3 #****
#         parameters_1['Electrolyte diffusivity [m2.s-1]'] = 5.35e-10 #****
#         parameters_1['Positive electrode diffusivity [m2.s-1]']: 4e-15
#         parameters_1['Negative electrode diffusivity [m2.s-1]']: 3.3e-14,
import utils_eps_orig as ute
lb = [0.13,5.35e-11,4e-16,3.3e-15] # lower bound on thetas
ub = [13,  5.35e-9, 4e-14,3.3e-13] # upper bound on thetas
nParam = 4; # dimension of parameter space
nY = 100; # dimension of observations (voltage)
nIn = 100;
nOut = 100;
lb_eps = -np.ones((nY,))*0.3e-3 #+/- 0.3 mV accuracy
ub_eps = np.ones((nY,))*0.3e-3
reuse = True; # if true, share inner and outer thetas
d_vals = inputs

#%% randomly sampling from priors on theta
thetas_outer = ute.sample_prior(nOut,nParam,lb,ub,seed=3141)
if reuse==True:
    thetas_inner = thetas_outer

#%% evaluating g(\theta,d) for inner and outer loops
    
batCap = 4.9872
d = np.array([-batCap/4, batCap/5, -batCap/6, batCap/7, -batCap/8, batCap/9, -batCap/10, 0.55, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -batCap/4])

nD = d.shape[0] # dimension of experimental design vector
inputs_outer = np.empty((nOut,nParam+nD))
for i in range(nOut):
    inputs_outer[i,0:nD] = d
    inputs_outer[i,nD:] = thetas_outer[i,:]
g_outer = pybamm_SPMe_Sim(inputs_outer)[0]
#g_outer=np.load("test_inners.npy")
if reuse==True:
    g_inner = g_outer
else:
    # fill this in later if we ever want to not use reuse
    print("use reuse instead pls")
    
#%% evaluating U(d)

eps_mean=np.ones((nY,)); eps_cov = np.diag(np.ones(nY,)*3e-3**2)
eps_outer = ute.sample_epsilon(nOut, nY, mean=eps_mean, cov=eps_cov)
uD = ute.eig_eps_fast(eps_outer,nOut,nIn,g_inner,g_outer,eps_mean,eps_cov)

    



