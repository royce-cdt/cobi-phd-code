if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from scipy.optimize import curve_fit
    from concurrent.futures import ThreadPoolExecutor

    output_log = '/rds/user/cja69/hpc-work/python/cl_fit.log'

    with open(output_log, 'w') as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"CL fit script started at {current_time}\n")

    #filepath = 'C:\\Users\\cobia\\OneDrive - University of Cambridge\\CL\\COBI-20250804\\HYP-SHORTEND09-REDO700\\HYPCard_corrected.hspy'
    # filepath = 'C:\\Users\\cobia\\OneDrive - University of Cambridge\\CL\\COBI20250707\\HYP-LONGEND00\\HYPCard_corrected.hspy'
    filepath = '/rds/user/cja69/hpc-work/python/HYPCard_corrected.npy'
    axis_value_file = '/rds/user/cja69/hpc-work/python/axis_values.npy'

    # m_si = cl_sem_eV_si.create_model()
    # bkg_si = hs.model.components1D.Offset()
    # g1_si = hs.model.components1D.GaussianHF()
    # m_si.extend([g1_si, bkg_si])

    # cl_sem_eV_si.axes_manager.indices = (5, 5)  # test pixel
    # g1_si.centre.value = 3.4
    # g1_si.fwhm.value = 0.1
    # g1_si.height.value = 500
    # bkg_si.offset.value = 200
    # g1_si.centre.bmax = g1_si.centre.value + 0.2
    # g1_si.centre.bmin = g1_si.centre.value - 0.2
    # g1_si.fwhm.bmin = 0.01

    # m_si.fit()
    # m_si.print_current_values()


    # m_si.assign_current_values_to_all()

    # print("Starting multifit...")
    # m_si.multifit(bounded=True, show_progressbar=True)
    # m_centre = g1_si.centre.as_signal()
    # m_intensity = g1_si.height.as_signal()
    # m_fwhm = g1_si.fwhm.as_signal()

    # output_path = 'C:\\Users\\cobia\\OneDrive - University of Cambridge\\CL\\COBI20250707\\HYP-LONGEND00\\'
    # output_path = 'C:\\Users\\cobia\\OneDrive - University of Cambridge\\CL\\COBI-20250804\\HYP-SHORTEND09-REDO700\\'
    output_path = '/rds/user/cja69/hpc-work/python/'
    # m_centre.save(output_path + 'm_centre.hspy')
    # m_intensity.save(output_path + 'm_intensity.hspy')
    # m_fwhm.save(output_path + 'm_fwhm.hspy')

    # param_maps = np.array([m_centre.data, m_fwhm.data, m_intensity.data]).transpose((1, 2, 0))
    # print(param_maps.shape)

    def gaussian(x, center, fwhm, height, offset):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return height * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset



    # shape (256, 256, 1024)
    image = np.load(filepath)  # your data cube
    axis_values = np.load(axis_value_file)  # shape (1024,), non-uniform energy axis

    # reshape to (65536, 1024) for easier iteration
    pixels = image.reshape(-1, image.shape[-1])

    # bounds
    center_guess = 3.4
    fwhm_guess = 0.1
    height_guess = 500
    offset_guess = 200

    lower_bounds = [center_guess - 0.2, 0.01, 0, 0]
    upper_bounds = [center_guess + 0.2, 1.0, 1e5, 1e4]
    initial_guess = [center_guess, fwhm_guess, height_guess, offset_guess]


    def fit_spectrum(spectrum):
        try:
            popt, _ = curve_fit(
                gaussian,
                axis_values,
                spectrum,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=500
            )
        except RuntimeError:
            popt = [np.nan, np.nan, np.nan, np.nan]
        return popt

    from tqdm import tqdm

    results = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(fit_spectrum, pixels), total=pixels.shape[0]))

    # reshape result to (256, 256, 4)
    param_maps = np.array(results).reshape(cl_sem.axes_manager.navigation_axes[0].size, cl_sem.axes_manager.navigation_axes[1].size, 4)

    # Save parameter maps as csv

    
    pd.DataFrame(param_maps[:, :, 0]).to_csv(output_path + 'm_centre.csv', index=False)
    pd.DataFrame(param_maps[:, :, 1]).to_csv(output_path + 'm_fwhm.csv', index=False)
    pd.DataFrame(param_maps[:, :, 2]).to_csv(output_path + 'm_intensity.csv', index=False)

    with open(output_log, 'a') as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"CL fit script finished at {current_time}\n")
