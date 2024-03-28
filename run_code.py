#first let me declare all the necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pybamm
import scipy
from scipy.special import hermite

import time as tm

def run_battery_simulation(a_nmc, b_nmc, c_nmc, d_nmc, my_graphite_diff_parameter, sep_por, neg_por, pos_por, cap_dl_neg, p1, p2, p3, p4, p5, p6, p7, p8):

    def buildPhi(x, order):
        Phi = np.zeros((len(x), order+1))
        for o in range(order+1):
            for row in range(len(x)):
                Phi[row,o] = hermite(o)(x[row])
        return Phi

    def diff_model(x, coeffs, order=3):
        # parameters:
        # x - input to make predictions at, must be an array
        # coeffs - regression coefficients to use in hermite poly model
        phi = buildPhi(x, order)
        return np.exp(phi@coeffs)
        
    def my_nmc_diff(c_s_p, T,a,b,c0,d):
        from pybamm import Interpolant,constants, exp
        from pybamm import exp, constants, maximum, minimum
        """

        References
        ----------
        Look at Chueh paper to figure out how he does that

        Parameters
        ----------



        Returns
        -------
        :class:`pybamm.Symbol`
            Electrolyte diffusivity
        """

        #data being used (from Kang and Chueh's NMC GITT paper)
        # x = np.array([ 0.0, 0.10949999999999999, 0.219, 0.3285, 0.438, 0.439354839, 0.456129032, 0.478064516, 0.498709677, 0.501290323, 0.536129032, 0.54, 0.573548387, 0.57483871, 0.586451613, 0.594193548, 0.658709677, 0.671612903, 0.684516129, 0.721935484, 0.736129032, 0.756774194, 0.776129032, 0.781290323, 0.787741935, 0.792903226, 0.798064516, 0.803225806, 0.856129032, 0.870322581, 0.874193548, 0.881935484, 0.885806452, 0.923225806, 0.928387097, 0.93, 0.930967742, 0.932258065, 0.9475, 0.9650000000000001, 0.9825, 1.0 ])
        # D_nmc = 1e-4*np.array([ 4.40E-10, 4.40E-10, 4.40E-10, 4.40E-10, 4.40E-10, 4.40E-10, 9.65E-10, 6.07E-10, 1.17E-09, 2.53E-09, 2.01E-09, 9.82E-10, 5.55E-10, 2.82E-10, 1.43E-09, 4.64E-10, 2.31E-10, 6.75E-11, 3.08E-11, 5.85E-11, 1.51E-10, 6.75E-11, 3.49E-11, 7.12E-11, 5.07E-11, 4.99E-11, 1.22E-10, 4.02E-11, 7.93E-11, 7.93E-11, 6.18E-11, 2.01E-10, 5.75E-11, 1.13E-10, 9.82E-11, 1.31E-10, 1.31E-10, 9.82E-11, 1.31E-10, 1.31E-10, 1.31E-10, 1.31E-10 ])


        #print("Exponent:", exp(a*c_s_p**3+b*c_s_p**2+c0*c_s_p+d))
        #return 8e-15
        #reg_output = diff_model(c_s_p,np.array([d,c0,b,a]))
        reg_output = exp( 1*d + 2*c_s_p*c0 + ( 4*c_s_p**2 - 2 ) * b + ( 8*c_s_p**3 - 12*c_s_p)*a )
        return minimum(maximum(reg_output,3e-15),2.5e-13) #max and min values from Chueh paper, fitting is also from Chueh paper






    def my_graphite_diff(c_s_n,T,diff_param):

        from pybamm import Interpolant,constants, exp

        x = np.array([0,0.055763963, 0.080379776, 0.101036403, 0.119914264, 0.135205906,
                            0.149751613, 0.158702818, 0.167654023 , 0.183318631, 0.190032035,0.19898324,
                            0.216086435, 0.239263661, 0.257805443, 0.27666691,
                            0.279544083, 0.297446492, 0.312924617, 0.33101351, 0.350194664,
                            0.358986025, 0.378966393, 0.395909745, 0.409336552, 0.430755507,
                            0.438427968, 0.456777938, 0.475191845, 0.497058359, 0.509043028,
                            0.529058916, 0.553994415, 0.576052742, 0.585283671, 0.601787455,
                            0.616333163, 0.630878871, 0.64878128, 0.673397094, 0.699451493,
                            0.72262872, 0.747244533, 0.765146943, 0.776335949, 0.79647616,
                            0.810275934, 0.833399879, 0.843469985, 0.86712674, 0.892701611,
                            0.917317424, 0.941933237, 0.966549051, 0.979975858])

        D = 1e-4*np.array([
            9.62E-12,9.62E-12, 1.05E-11, 1.30E-11, 1.64E-11, 1.64E-11, 1.36E-10, 1.90E-10,2.43E-10,
            3.99E-10, 2.64E-10, 1.68E-10, 7.77E-11, 6.42E-11, 6.55E-11, 7.09E-11, 9.81E-11,
            6.82E-11, 1.39E-10, 1.83E-10, 2.23E-10, 3.36E-10, 2.28E-10, 2.16E-10, 3.62E-10,
            3.86E-10, 5.12E-10, 3.97E-10, 3.40E-10, 3.67E-10, 3.10E-10, 3.25E-10, 3.77E-10,
            3.15E-10, 2.68E-10, 2.29E-10, 4.27E-11, 2.56E-10, 3.58E-10, 3.63E-10, 3.56E-10,
            3.32E-10, 3.57E-10, 3.94E-10, 3.08E-10, 2.99E-10, 4.03E-10, 4.12E-10, 3.58E-10,
            3.34E-10, 3.42E-10, 3.25E-10, 3.29E-10, 3.56E-10, 3.63E-10
        ])




        D_m2s = Interpolant(x,D,c_s_n,interpolator="cubic") 
        
        return diff_param*D_m2s



    deg_options = {"surface form":"differential", "thermal": "lumped","cell geometry": "arbitrary","SEI":"reaction limited","SEI film resistance":"average","lithium plating":"reversible"} #haven't added mechanical fracture yet 
    default_options = {"thermal":"isothermal"}
    model_SPMe_1 = pybamm.lithium_ion.SPMe(options=deg_options)


    # Get the default solver for the model
    default_solver = model_SPMe_1.default_solver
    print("Default Solver:", default_solver.name)

    var_pts_1 = {
        "x_n": 50,  # negative electrode
        "x_s": 25,  # separator 
        "x_p": 50,  # positive electrode
        "r_n": 50,  # negative particle
        "r_p": 50,  # positive particle
    }

    parameters_1 = pybamm.ParameterValues("Chen2020")

    parameters_1['Cation transference number'] = 0.38
    parameters_1['Cell volume [m3]'] = 3.914e-5

    parameters_1['Electrolyte conductivity [S.m-1]'] = 1.3

    parameters_1['Negative particle radius [m]'] = 2.5e-6

    parameters_1['Positive particle radius [m]'] = 3.5e-6
    parameters_1['Separator porosity'] = sep_por
    parameters_1['Negative electrode porosity'] = neg_por #default value from Chen2020 is 0.25
    parameters_1['Positive electrode porosity'] = pos_por #default value from Chen2020 is 0.335 

    from parameter_settings import get_keys_and_values
    keys_and_values = get_keys_and_values(cap_dl_neg)


    parameters_1.update(keys_and_values, check_already_exists=False)

    parameters_1['Cation transference number'] = 0.38
    parameters_1['Cell volume [m3]'] = 3.914e-5
    parameters_1['Electrode height [m]'] = 0.4527
    parameters_1['Electrode width [m]'] = 0.4527
    parameters_1['Electrolyte conductivity [S.m-1]'] = 1.3

    parameters_1['Maximum concentration in negative electrode [mol.m-3]'] = 28746
    parameters_1['Maximum concentration in positive electrode [mol.m-3]'] = 35380


    #use the following to play with SOC ->
    parameters_1['Initial concentration in negative electrode [mol.m-3]'] = 2392.8+11000
    parameters_1['Initial concentration in positive electrode [mol.m-3]'] = 28588-11000
    parameters_1['Nominal cell capacity [A.h]'] = 4.9872 #this doesn't really matter as a parameter - see documentation/github https://github.com/pybamm-team/PyBaMM/discussions/1635#discussioncomment-1261662

    #for positive electrode -

    parameters_1['Positive electrode diffusivity [m2.s-1]'] = lambda c_s_p, T: my_nmc_diff(c_s_p,T,a_nmc,b_nmc,c_nmc,d_nmc) #default was 8e-15

    #for negative electrode -
    parameters_1['Negative electrode diffusivity [m2.s-1]'] = lambda c_s_n, T: my_graphite_diff(c_s_n, T, my_graphite_diff_parameter) #default was 5e-15






    def create_custom_experiment(p1,p2,p3,p4,p5,p6,p7,p8):


        def my_fun(A, constant):
                
                def current(t):
                    sine_terms = A * pybamm.sin(2 * np.pi * t) 
                    return sine_terms + constant
                return current
        
        t = np.linspace(0, 20, 6000) #change this time to be one period of the smallest sine (or a quarter of a period) 

        def chirp_signal(f0=0.0,t1=20): #look at p7 and p8 and fit them - t1 was 1 previously
            chirp = scipy.signal.chirp(t,f0,t1,1000)
            return chirp




        drive_cycle_power = np.column_stack([t, chirp_signal(p7,p8)])

        experiment = pybamm.Experiment( [pybamm.step.current(drive_cycle_power)]+[pybamm.step.current(drive_cycle_power),
            ("Rest for 200 seconds", f"Charge at {p1}C for {p2} seconds or until 4.2 V", f"Rest for {p3} seconds", f"Charge at {p4}C for {p5} seconds or until 4.2 V", f"Rest for {p6} seconds", f"Discharge at {p4}C for {p5} seconds or until 3.0 V", "Rest for 300 seconds")
        ] * 5 )
        experiment_ending = my_fun(5,0.1)
        return experiment


    custom_experiment = create_custom_experiment(p1,p2,p3,p4,p5,p6,p7,p8)

    start_time = tm.time()
    experiment = pybamm.Experiment(
        [("Rest for 2 minutes","Charge at 1 C for 12 minutes or until 4.2 V","Rest for 1 hour","Discharge at C/3 for 12 minutes or until 3.0 V","Rest for 1 hour")] 
    )



    initial_voltage = 3.9 
    sim_1 = pybamm.Simulation(model_SPMe_1, experiment=custom_experiment,parameter_values=parameters_1, var_pts=var_pts_1)
    sim_1.solve([0,150])
    end_time = tm.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    #return sim_1.solution
    time = sim_1.solution["Time [s]"].data
    terminal_voltage = sim_1.solution['Terminal voltage [V]'].data
    return time, terminal_voltage
