# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, welch
import os
from scipy.optimize import curve_fit
import time


def load_files(integer):
    folder_path = "simulation_results_fixed_design"
    file_extension = f"_{integer}.npy"
    date_time_str = ""

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            # Extract the date and time part from the filename
            parts = filename.split('_')
            date_time_str = f"{parts[1]}_{parts[2]}_{parts[3].split('.')[0]}"

            if filename.startswith("params"):
                theta = np.load(os.path.join(folder_path, filename))
            elif filename.startswith("time_data"):
                t = np.load(os.path.join(folder_path, filename))
            elif filename.startswith("voltage_data"):
                v = np.load(os.path.join(folder_path, filename))
            elif filename.startswith("current_data"):
                current = np.load(os.path.join(folder_path, filename))

    return theta, t, v, current, date_time_str

nIntegers = 50
runCount = 0
for integer in range(nIntegers):
    # Load files based on the provided integer
    theta, t, v_orig, current, date_time_str = load_files(integer)

    # add noise to signal for likelihood estimation
    nEps = 30 # number of MC samples of epsilon to propagate
    totalStats = 209
    summary_stats_all = np.zeros((totalStats, nEps*nIntegers))
    for epsCount in range(nEps):   
        eps_all = np.random.normal(loc=0.0, scale=3e-3, size=len(v_orig)) 
        v = v_orig + eps_all
        p1, p2, p3, p4, p5, p6, p7, p8 = theta[9:17]

        start_time=time.time()

        # Define block durations and repeat to create the full durations
        block_durations_sizing = [p8 + 200, p2 + p3, p5 + p6, p5 + 300]
        full_durations_sizing = np.tile(block_durations_sizing, 5)  # Adjust the repetition count as needed

        # Calculate start and end times for each block
        start_times_sizing = np.cumsum(np.concatenate(([0], full_durations_sizing[:-1])))
        end_times_sizing = start_times_sizing + full_durations_sizing

        # Find indices for start and end times using vectorized operations
        start_indices_sizing = np.searchsorted(t, start_times_sizing, side='right')
        end_indices_sizing = np.searchsorted(t, end_times_sizing, side='right') - 1  # -1 because end index is inclusive

        # Explicitly set the start of the first block to 0
        start_indices_sizing[0] = 0

        # Ensure the end of the last block does not exceed the length of the array
        end_indices_sizing[-1] = min(end_indices_sizing[-1], len(t) - 1)

        # Create the blocks based on the indices found
        tBlocks_sizing = [t[start:end+1] for start, end in zip(start_indices_sizing, end_indices_sizing)]
        vBlocks_sizing = [v[start:end+1] for start, end in zip(start_indices_sizing, end_indices_sizing)]
        iBlocks_sizing = [current[start:end+1] for start, end in zip(start_indices_sizing, end_indices_sizing)]

        # Find the largest block size
        largest_block_size = max(len(block) for block in tBlocks_sizing)

        num_points_chirp = largest_block_size

        num_points = num_points_chirp*40
        #num_points = len(t)
        t_interp = np.linspace(t[0], t[-1], num_points)

        v_interp_func = interp1d(t, v, kind='linear')
        i_interp_func = interp1d(t, current, kind='linear')

        v_interp = v_interp_func(t_interp)
        i_interp = i_interp_func(t_interp)

        # Break the interpolated data into blocks
        block_durations = [p8, 200, p2, p3, p5, p6, p5, 300]
        full_durations = block_durations * 5  

        start_times = [0]
        for duration in full_durations[:-1]: 
            next_start_time = start_times[-1] + duration
            start_times.append(next_start_time)
        end_times = [start + duration for start, duration in zip(start_times, full_durations)]

        def find_index(time_array, time):
            return np.searchsorted(time_array, time, side='right')

        start_indices = [find_index(t_interp, time) for time in start_times]
        end_indices = [find_index(t_interp, time) - 1 for time in end_times] 


        start_indices[0] = 0

        tBlocks = [t_interp[start:end+1] for start, end in zip(start_indices, end_indices)]
        vBlocks = [v_interp[start:end+1] for start, end in zip(start_indices, end_indices)]
        iBlocks = [i_interp[start:end+1] for start, end in zip(start_indices, end_indices)]

        interp_tBlocks = tBlocks
        interp_vBlocks = vBlocks
        interp_iBlocks = iBlocks


        def summarize_peak_time_differences(time_uniform, output_voltage_uniform):
            positive_peaks, _ = find_peaks(output_voltage_uniform)
            negative_peaks, _ = find_peaks(-output_voltage_uniform)
            all_peaks = np.sort(np.concatenate((positive_peaks, negative_peaks)))
            time_differences = []
            paired_peaks = []

            for i in range(len(all_peaks) - 1):
                if (all_peaks[i] in positive_peaks and all_peaks[i+1] in negative_peaks) or \
                (all_peaks[i] in negative_peaks and all_peaks[i+1] in positive_peaks):
                    time_diff = time_uniform[all_peaks[i+1]] - time_uniform[all_peaks[i]]
                    time_differences.append(time_diff)
                    paired_peaks.append((all_peaks[i], all_peaks[i+1]))

            max_time_diff = np.max(time_differences) if time_differences else np.nan
            min_time_diff = np.min(time_differences) if time_differences else np.nan
            mean_time_diff = np.mean(time_differences) if time_differences else np.nan
            std_time_diff = np.std(time_differences) if time_differences else np.nan

            return max_time_diff, min_time_diff, mean_time_diff, std_time_diff

        def summarize_impedance_phase(frequency_ranges, impedance, phase_shift, frequencies):
            def average_and_std_over_frequency_range(frequencies, data, frequency_ranges):
                averages = []
                stds = []
                for f_min, f_max in frequency_ranges:
                    mask = (frequencies >= f_min) & (frequencies < f_max)
                    avg = np.mean(data[mask])
                    std = np.std(data[mask])
                    averages.append(avg)
                    stds.append(std)
                return averages, stds

            average_impedance, std_impedance = average_and_std_over_frequency_range(frequencies, impedance, frequency_ranges)
            average_phase, std_phase = average_and_std_over_frequency_range(frequencies, phase_shift, frequency_ranges)

            return average_impedance, std_impedance, average_phase, std_phase

        def calculate_power_and_bandwidth(output_voltage_uniform, time_uniform, percentage=0.95):
            sampling_rate = 1 / (time_uniform[1] - time_uniform[0])
            frequencies, psd = welch(output_voltage_uniform, fs=sampling_rate)
            total_power = np.trapz(psd, frequencies)

            def power_bandwidth(frequencies, psd, percentage=0.95):
                cumulative_power = np.cumsum(psd) / np.sum(psd)
                lower_index = np.where(cumulative_power >= (1 - percentage) / 2)[0][0]
                upper_index = np.where(cumulative_power <= 1 - (1 - percentage) / 2)[0][-1]
                bandwidth = frequencies[upper_index] - frequencies[lower_index]
                return frequencies[lower_index], frequencies[upper_index], bandwidth

            lower_freq, upper_freq, bandwidth = power_bandwidth(frequencies, psd, percentage=percentage)

            return total_power, lower_freq, upper_freq, bandwidth

        def calculate_impedance_and_phase(time, input_current, output_voltage):
            n = len(time)
            d = time[1] - time[0]
            input_fft = fft(-input_current)
            voltage_fft = fft(output_voltage)
            frequencies = fftfreq(n, d)
            impedance = np.abs(voltage_fft / input_fft)
            phase_shift = np.angle(voltage_fft) - np.angle(input_fft)
            return frequencies, impedance, phase_shift

        def combined_summary_chirp(time_uniform, input_current_uniform, output_voltage_uniform, frequency_ranges, percentage=0.95):
            max_time_diff, min_time_diff, mean_time_diff, std_time_diff = summarize_peak_time_differences(time_uniform, output_voltage_uniform)
            frequencies, impedance, phase_shift = calculate_impedance_and_phase(time_uniform, input_current_uniform, output_voltage_uniform)
            average_impedance, std_impedance, average_phase, std_phase = summarize_impedance_phase(frequency_ranges, impedance, phase_shift, frequencies)
            total_power, lower_freq, upper_freq, bandwidth = calculate_power_and_bandwidth(output_voltage_uniform, time_uniform, percentage)

            results = {
                "peak_time_differences": {
                    "max_time_diff": max_time_diff,
                    "min_time_diff": min_time_diff,
                    "mean_time_diff": mean_time_diff,
                    "std_time_diff": std_time_diff
                },
                "impedance_phase_summary": {
                    "average_impedance": average_impedance,
                    "std_impedance": std_impedance,
                    "average_phase": average_phase,
                    "std_phase": std_phase
                },
                "power_bandwidth_summary": {
                    "total_power": total_power,
                    "lower_freq": lower_freq,
                    "upper_freq": upper_freq,
                    "bandwidth": bandwidth
                }
            }

            return results

        # for block % 8 == 1
        def exponential_decay(t, k):
            return np.exp(-k * t)
        def negative_exponential_decay(t, k):
            return -np.exp(-k * t)
        def process_special_block(v_block, t_block):
            # First voltage value (V0) is just the first value in v_block
            # Perform the curve fitting
            # def fit_exponential_decay(t, V):
            #     V0 = V[0]
            #     t_values = t - t[0]  # Time differences relative to the first time point
            #     v_values = V / V0  # Normalize by V0
            #     popt, _ = curve_fit(exponential_decay, t_values, v_values)
            #     k = popt[0]
            #     return V0, k
            # V0, k = fit_exponential_decay(t_block,v_block)
            V0 = v_block[0]

            # Fit the difference to an exponential decay V = V0 * exp(-k * t) to find k
            #t_values = t_block[:2] - t_block[0]  # Time differences relative to the first time point
            #v_values = v_block[:2] / V0  # Normalize by V0
            #popt, _ = curve_fit(exponential_decay, t_values, v_values)
            #k = popt[1]
            #print("popt length", len(popt))
            #print("popt:", popt)
            #print("vblock [1]", v_block[1])
            #print("vblock[0]", v_block[0])
            k = ( (v_block[1] - v_block[0]) / (t_block[1] - t_block[0]) )
            k = k / V0#because when you linearize the slope has a V0 term to it 
            #print("k", k)

            # Find the final voltage value in the block
            final_V = v_block[-1]

            return V0, k, final_V

        def process_exponential_decay_block(v_block, t_block):
            #num_points = int(np.ceil(len(v_block) / 4))
            #print("This is for the block 3 calculation")
            #def fit_exponential_decay(t, V, is_positive_decay):
                #V0 = V[0]
                #t_values = t - t[0]  # Time differences relative to the first time point
                #v_values = V / V0  # Normalize by V0
                #t_values = t - t[0]
                #v_values = V
                #popt, _ = curve_fit(exponential_decay, t_values, v_values)
                #initial_guess = [1e-5]
                #bounds = (0, 1)
                #if is_positive_decay:
                #    popt, pcov = curve_fit(exponential_decay, t_values, v_values, p0=initial_guess, bounds=bounds)
                #else:
                #    popt, pcov = curve_fit(negative_exponential_decay, t_values, v_values, p0=initial_guess, bounds=bounds)

                #popt, pcov = curve_fit(exponential_decay, t_values, v_values, p0=initial_guess, bounds=bounds)
                #k = popt[0]
                #return k, pcov

            #print("This is the t_values:", )
            V0 = v_block[0]
            #t_values = t_block[:num_points]  # Time differences relative to the first time point
            #print("This is the t_values:", t_values)
            #v_values = v_block[:num_points]
            final_V = v_block[-1]
            is_positive_decay = final_V < V0 
            #print("This is the v_values:", v_values)
            #k, pcov = fit_exponential_decay(t_values,v_values, is_positive_decay)
            #popt, _ = curve_fit(exponential_decay, t_values, v_values)
            #k = popt[0]

            return V0, k, final_V

        def calculate_summary_statistics(t_block, v_block, i_block):
            # Calculate resistance
            resistance = np.abs( (v_block[0] - v_block[-1]) / np.mean(i_block))

            # Calculate dV/dt
            dV_dt = np.gradient(v_block, t_block)
            #dV_dt = np.mean(dV_dt)
            # Calculate charge (Q) by integrating current over time
            Q = np.cumsum(-i_block) * (t_block[1] - t_block[0])

            # Calculate dV/dQ
            dV_dQ = np.gradient(v_block, Q)
            #dV_dQ = np.mean(dV_dQ)

            return resistance, np.mean(dV_dt), np.mean(dV_dQ), Q
        
        frequency_ranges = [(1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1), (1, 10)]


        results_blocks = []
        results_array = []


        for blockNumber, (t_block, v_block, i_block) in enumerate(zip(interp_tBlocks, interp_vBlocks, interp_iBlocks)):
            if blockNumber % 8 == 0:
                results = combined_summary_chirp(t_block, i_block, v_block, frequency_ranges)
                results_blocks.append((blockNumber, results))
                results_array.append(results)
            elif blockNumber % 8 == 1:
                V0, k, final_V = process_special_block(v_block, t_block)
                result = {"V0": V0, "k": k, "final_V": final_V}
                results_blocks.append((blockNumber, result))
                results_array.append(result)
            elif blockNumber % 8 == 2:
                resistance, dV_dt, dV_dQ, Q = calculate_summary_statistics(t_block, v_block, i_block)
                result = {
                    "resistance": resistance,
                    "dV_dt": dV_dt,
                    "dV_dQ": dV_dQ
                }
                results_blocks.append((blockNumber, result))
                results_array.append(result)
            elif blockNumber % 8 == 3:
                V0, k, final_V = process_special_block(v_block, t_block)
                result = {"V0": V0, "k": k, "final_V": final_V}
                results_blocks.append((blockNumber, result))
                results_array.append(result)
            elif blockNumber % 8 == 4:
                resistance, dV_dt, dV_dQ, Q = calculate_summary_statistics(t_block, v_block, i_block)
                result = {
                    "resistance": resistance,
                    "dV_dt": dV_dt,
                    "dV_dQ": dV_dQ
                }
                results_blocks.append((blockNumber, result))
                results_array.append(result)
            elif blockNumber % 8 == 5:
                V0, k, final_V = process_special_block(v_block, t_block)
                result = {"V0": V0, "k": k, "final_V": final_V}
                results_blocks.append((blockNumber, result))
                results_array.append(result)
            elif blockNumber % 8 == 6:
                resistance, dV_dt, dV_dQ, Q = calculate_summary_statistics(t_block, v_block, i_block)
                result = {
                    "resistance": resistance,
                    "dV_dt": dV_dt,
                    "dV_dQ": dV_dQ
                }
                results_blocks.append((blockNumber, result))
                results_array.append(result)
            elif blockNumber % 8 == 7:
                V0, k, final_V = process_special_block(v_block, t_block)
                result = {"V0": V0, "k": k, "final_V": final_V}
                results_blocks.append((blockNumber, result))
                results_array.append(result)

        results_array = []


        nu_array = []
        for blockNumber, results in results_blocks:
            if blockNumber % 8 == 0:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print("Peak Time Differences Summary:")
                for key, value in results["peak_time_differences"].items():
                    #print(f"  {key.replace('_', ' ').title()}: {value:.3f} s")
                    nu_array.append(value)

                #print("Impedance and Phase Summary:")
                for (f_min, f_max), avg_imp, std_imp, avg_phase, std_phas in zip(frequency_ranges, results["impedance_phase_summary"]["average_impedance"], results["impedance_phase_summary"]["std_impedance"], results["impedance_phase_summary"]["average_phase"], results["impedance_phase_summary"]["std_phase"]):
                    #print(f"  Frequency range {f_min} to {f_max} Hz:")
                    #print(f"    Average Impedance: {avg_imp:.3f} Ohms")
                    nu_array.append(avg_imp)
                    #print(f"    Std Impedance: {std_imp:.3f} Ohms")
                    nu_array.append(std_imp)
                    #print(f"    Average Phase: {avg_phase:.3f} Radians")
                    nu_array.append(avg_phase)
                #   print(f"    Std Phase: {std_phas:.3f} Radians")
                    nu_array.append(std_phas)
                    #print(f"This is block: {blockNumber}")

            # print("Power and Bandwidth Summary:")
                for key, value in results["power_bandwidth_summary"].items():
                #    print(f"  {key.replace('_', ' ').title()}: {value:.3f} Hz")
                    nu_array.append(value)


                values = [
                    results["peak_time_differences"]["max_time_diff"],
                    results["peak_time_differences"]["min_time_diff"],
                    results["peak_time_differences"]["mean_time_diff"],
                    results["peak_time_differences"]["std_time_diff"],
                    *results["impedance_phase_summary"]["average_impedance"],
                    *results["impedance_phase_summary"]["std_impedance"],
                    *results["impedance_phase_summary"]["average_phase"],
                    *results["impedance_phase_summary"]["std_phase"],
                    results["power_bandwidth_summary"]["total_power"],
                    results["power_bandwidth_summary"]["lower_freq"],
                    results["power_bandwidth_summary"]["upper_freq"],
                    results["power_bandwidth_summary"]["bandwidth"]
                ]
                #results_array.append(results["peak_time_differences"]["max_time_diff"])
            elif blockNumber % 8 == 1:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print(f"  V0: {results['V0']:.3f}")
                nu_array.append(results['V0'])
                nu_array.append(results['k'])
                nu_array.append(results['final_V'])
                #print(f"  k: {results['k']:.9f}")
                #print(f"  Final V: {results['final_V']:.3f}")
                values = [results['V0'], results['k'], results['final_V']]
                #print(f"This is block: {blockNumber}")
            elif blockNumber % 8 == 2:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print(f"  Resistance: {results['resistance']:.3f} Ohms")
                #print(f"  dV/dt: {results['dV_dt']}")
                #print(f"  dV/dQ: {results['dV_dQ']}")
                values = [results['resistance'], results['dV_dt'], results['dV_dQ']]
                nu_array.append(results['resistance'])
                nu_array.append(results['dV_dt'])
                nu_array.append(results['dV_dQ'])
                #print(f"This is block: {blockNumber}")
            elif blockNumber % 8 == 3:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print(f"  V0: {results['V0']:.3f}")
                #print(f"  k: {results['k']:.9f}")
                #print(f"  Final V: {results['final_V']:.3f}")
                values = [results['V0'], results['k'], results['final_V']]
                nu_array.append(results['V0'])
                nu_array.append(results['k'])
                nu_array.append(results['final_V'])
                #print(f"This is block: {blockNumber}")
            elif blockNumber % 8 == 4:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print(f"  Resistance: {results['resistance']:.3f} Ohms")
                #print(f"  dV/dt: {results['dV_dt']}")
                #print(f"  dV/dQ: {results['dV_dQ']}")
                values = [results['resistance'], results['dV_dt'], results['dV_dQ']]
                nu_array.append(results['resistance'])
                nu_array.append(results['dV_dt'])
                nu_array.append(results['dV_dQ'])
                #print(f"This is block: {blockNumber}")
            elif blockNumber % 8 == 5:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print(f"  V0: {results['V0']:.3f}")
                #print(f"  k: {results['k']:.9f}")
                #print(f"  Final V: {results['final_V']:.3f}")
                values = [results['V0'], results['k'], results['final_V']]
                nu_array.append(results['V0'])
                nu_array.append(results['k'])
                nu_array.append(results['final_V'])
                #print(f"This is block: {blockNumber}")
            elif blockNumber % 8 == 6:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print(f"  Resistance: {results['resistance']:.3f} Ohms")
                #print(f"  dV/dt: {results['dV_dt']}")
                #print(f"  dV/dQ: {results['dV_dQ']}")
                values = [results['resistance'], results['dV_dt'], results['dV_dQ']]
                nu_array.append(results['resistance'])
                nu_array.append(results['dV_dt'])
                nu_array.append(results['dV_dQ'])
                #print(f"This is block: {blockNumber}")
            elif blockNumber % 8 == 7:
                #print(f"This is block: {blockNumber}")

                #print(f"Block Number: {blockNumber}")
                #print(f"  V0: {results['V0']:.3f}")
                #print(f"  k: {results['k']:.15f}")
                #print(f"  Final V: {results['final_V']:.3f}")
                nu_array.append(results['V0'])
                nu_array.append(results['k'])
                nu_array.append(results['final_V'])
                values = [results['V0'], results['k'], results['final_V']]
                #print(f"This is block: {blockNumber}")


        nu_array_np = np.array(nu_array)
        nanIndices = [5, 7, 50, 52, 94,  95,  96,  97, 139, 140, 141, 142, 184, 185, 186, 187]
        nu_array_np = np.delete(nu_array_np, nanIndices)
        # Print the total number of summary statistics
        #print(f"Total number of summary statistics stored in results_blocks: {len[nu_array_np]}")
        summary_stats_all[:, runCount] = nu_array_np
        runCount += 1

    # nu_array_np = np.array(nu_array)
    # folder_path = "summary_statistics_fixed_design"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # # Define the new file name based on the extracted date, time, and integer value
    # file_name = f"summary_stats_{date_time_str}_{integer}.npy"
    # file_path = os.path.join(folder_path, file_name)

    # # Save the numpy array to the file
    # np.save(file_path, nu_array_np)

    # print(f"Array saved to {file_path}")

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time for integer {integer}: {elapsed_time} seconds")




# %%
