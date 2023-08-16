import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as tm 
import random
from bayes_opt import BayesianOptimization
from multiprocessing import Pool
import multiprocessing
import datetime
from concurrent.futures import ThreadPoolExecutor

def my_fun(A_values, B_values, omega_values, constant):
    def current(t):
        sine_terms = np.sum([A * pybamm.sin(2 * np.pi * omega * t) for A, omega in zip(A_values, omega_values)], axis=0)
        cosine_terms = np.sum([B * pybamm.cos(2 * np.pi * omega * t) for B, omega in zip(B_values, omega_values)], axis=0)
        return sine_terms + cosine_terms + constant

    return current

def process_input(i, inputs, num_inputs, model_SPMe_1, parameters_1, var_pts_1):
    print("running for input #" + str(i) + " of " + str(num_inputs))
    k = 4
    A_values = inputs[i, :k]
    B_values = inputs[i, k:2*k]
    omega_values = np.linspace(1,k,num=k)/12.0 #omega_values = inputs[i, 8:12]
    constant = inputs[i, 2*k]

    parameters_1['Current function [A]'] = my_fun(A_values, B_values, omega_values, constant)

    thetas = inputs[i, 2*k:]
    
    parameters_1['Electrolyte conductivity [S.m-1]'] = thetas[0]#1.3 #****
    parameters_1['Electrolyte diffusivity [m2.s-1]'] = thetas[1]#5.35e-10 #****
    parameters_1['Negative electrode diffusivity [m2.s-1]'] = thetas[2]
    parameters_1['Positive electrode diffusivity [m2.s-1]'] = thetas[3]
    
    parameters_1['Cation transference number'] = 0.38
    parameters_1['Cell volume [m3]'] = 3.914e-5
    parameters_1['Electrode height [m]'] = 0.4527
    parameters_1['Electrode width [m]'] = 0.4527
    parameters_1['Initial concentration in negative electrode [mol.m-3]'] = 2392.8
    parameters_1['Initial concentration in positive electrode [mol.m-3]'] = 28595
    parameters_1['Maximum concentration in negative electrode [mol.m-3]'] = 28746
    parameters_1['Maximum concentration in positive electrode [mol.m-3]'] = 35380
    parameters_1['Negative electrode thickness [m]'] = 6.2e-5
    parameters_1['Negative particle radius [m]'] = 2.5e-6
    parameters_1['Nominal cell capacity [A.h]'] = 4.9872
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

    return terminal_voltage, currentData, time_vals

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

    num_inputs = inputs.shape[0]

    terminal_voltage_list = []
    currentData_list = []
    time_vals_list = []

    start_time = tm.time()

    import os
    with Pool(os. cpu_count()-1) as pool:
        # Pass num_inputs as an argument to process_input function using functools.partial
        import functools
        func = functools.partial(process_input, inputs=inputs, num_inputs=num_inputs, model_SPMe_1 = model_SPMe_1, parameters_1 = parameters_1, var_pts_1 = var_pts_1)
        results = pool.map(func, range(num_inputs))

    for result in results:
        terminal_voltage, currentData, time_vals = result
        terminal_voltage_list.append(terminal_voltage)
        currentData_list.append(currentData)
        time_vals_list.append(time_vals)

    end_time = tm.time()
    execution_time = end_time - start_time
    print(f"Execution time for {num_inputs} input tuples: {execution_time:.2f} seconds")
    
    plot_bool = False #boolean to decide plotting
    if plot_bool: # perhaps we add some plotting Bool here
        for i in [0]:#range(len(time_vals_list)):
            terminal_voltage = terminal_voltage_list[i]
            currentData = currentData_list[i]
            time_vals = time_vals_list[i]

            fig, ax1 = plt.subplots()
            ax1.plot(time_vals, terminal_voltage, label='Terminal Voltage', color='blue')
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Terminal Voltage [V]', color='blue')
            ax1.tick_params('y', colors='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(time_vals, currentData, label='Current', color='orange')
            ax2.set_ylabel('Current [A]', color='orange')
            ax2.tick_params('y', colors='orange')
            
            plt.title(f'Terminal Voltage and Current for {i+1}th element in input list')
            plt.legend()
            plt.show()

    return terminal_voltage_list, currentData_list, time_vals_list
     

def eig_reuse(d,nOut,nParam,nY):

    thetas_outer = ute.sample_prior(nOut,nParam,lb,ub,seed=3143)

    nD = d.shape[0] # dimension of experimental design vector
    inputs_outer = np.empty((nOut,nParam+nD),dtype=object)
    for i in range(nOut):
        inputs_outer[i,0:nD] = d
        inputs_outer[i,nD:] = thetas_outer[i,:]
    g_outer = pybamm_SPMe_Sim(inputs_outer)[0]#np.load("g_outer_broken.npy",allow_pickle=True)#pybamm_SPMe_Sim(inputs_outer)[0]
    #np.save("g_outer_broken.npy",g_outer)
    g_inner = g_outer
    
    nY = min(len(array) for array in g_outer)
    for i in range(nOut):
        g_outer[i] = g_outer[i][:nY]

    eps_mean=np.zeros((nY,)); eps_cov = np.diag(np.ones(nY,)*3e-3**2)   
    eps_outer = ute.sample_epsilon(nOut, nY, mean=eps_mean, cov=eps_cov,seed=1)
    uD = ute.eig_eps_fast_nd(eps_outer,nOut,nIn,g_inner,g_outer,eps_mean,eps_cov)

    return uD



if __name__ == "__main__":
    start_time_total = tm.time()

    import utils_eps_orig as ute
    lb = [0.13,5.35e-11,4e-16,3.3e-15] # lower bound on thetas
    ub = [13,  5.35e-9, 4e-14,3.3e-13] # upper bound on thetas
    nParam = 4; # dimension of parameter space
    nY = 100; # dimension of observations (voltage)

    # Add a dialog box to input nIn
    nIn = input("Please enter the value of nIn: ")

    # Convert the input to an integer (assuming nIn is an integer)
    nIn = int(nIn)

    # Set nOut equal to nIn
    nOut = nIn
    """How is the input array d defined? The first 4 inputs are the sine amplitudes, the next 4 are the cosine amplitudes, 
    and the last term is the constant. The equation for current becomes -> 
    \text{current}(t) = \sum_{i=1}^{k} A_{\text{values}_i} \cdot \sin(2 \pi \cdot \omega_{\text{values}_i} \cdot t) + \sum_{i=1}^{k} B_{\text{values}_i} \cdot \cos(2 \pi \cdot \omega_{\text{values}_i} \cdot t) + \text{constant}    
    """    
    batCap = 4.9872
    #d = np.array([-batCap/4, batCap/5, -batCap/6, batCap/7, batCap/4, -batCap/5, batCap/6, batCap/9 , -batCap/4])
    # This is an example of a case where nY != 100, (nY=89 here)
    d= np.array([-4.9872, 4.9872, 4.9872, -4.9872, 4.9872, 4.9872, 4.9872, 1.5903905102732894, 4.9872])
    # This is the optimal design from nOut=100 BayesOpt result
    #d=np.array([-4.9872,-2.2541241421607903,4.9872, -0.17756030687109964, -0.13824609495841444, 0.2011070930706761, 4.9872, 4.9872, 4.9872])
    #d=np.array([-4.9872/10,-4.9872/10,4.9872/10, -4.9872/10, -4.9872/10, 4.9872/10, 4.9872/10, 4.9872/10, 4.9872/10])
    uD = eig_reuse(d,nOut,nParam,nY)
    print(uD)

    def bo_friendly(d0,d1,d2,d3,d4,d5,d6,d7,d8,nOut,nParam,nY):
        d = np.array([d0,d1,d2,d3,d4,d5,d6,d7,d8],dtype=object)
        u_d = eig_reuse(d,nOut,nParam,nY)
        return u_d

    pbounds={'d0': (-batCap,batCap),'d1': (-batCap,batCap),'d2': (-batCap,batCap),'d3': (-batCap,batCap), \
            'd4':(-batCap,batCap), 'd5':(-batCap,batCap), 'd6':(-batCap,batCap), 'd7':(-batCap,batCap), \
            'd8': (-batCap,batCap)}

    optimizer = BayesianOptimization(
        f=lambda d0,d1,d2,d3,d4,d5,d6,d7,d8: bo_friendly(d0,d1,d2,d3,d4,d5,d6,d7,d8,nOut,nParam,nY), 
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    #optimizer.maximize(
    #    init_points=1,
    #    n_iter=50,
    #)
    #print(optimizer.max)

    print("Done")
    end_time_total = tm.time()
    execution_time_total = end_time_total - start_time_total

    # Convert execution_time_total to minutes
    execution_time_hours = execution_time_total / 60.0


    print(f"Execution time for {nIn} input tuples: {execution_time_hours:.2f} minutes")

    # Get the current system time and date
    current_time = datetime.datetime.now().strftime("%m/%d/%y %H:%M:%S")

    # Save the runtime to a new .txt file
    with open("shell_runtime_thomas_small_ds_parallel_multiprocessing_pool.txt", "a") as file:
        file.write(f"Date and time: {current_time}, Execution time for {nIn} input tuples: {execution_time_hours:.2f} minutes\n")

    print("Done")
