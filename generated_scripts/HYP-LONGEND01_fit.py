if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    import numpy as np
    from datetime import datetime
    from scipy.optimize import curve_fit
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    filepath = "/rds/user/cja69/hpc-work/python/cl_data/HYP-LONGEND01_CL_SEM.npy"
    axis_value_file = "/rds/user/cja69/hpc-work/python/cl_data/HYP-LONGEND01_axis.npy"
    output_log = "/rds/user/cja69/hpc-work/python/cl_data/HYP-LONGEND01_cl_fit.log"
    output_path = "/rds/user/cja69/hpc-work/python/cl_data/HYP-LONGEND01"

    with open(output_log, 'w') as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"CL fit script started at {current_time}\n")

    def gaussian(x, center, fwhm, height, offset):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return height * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset

    # shape (256, 256, 1024)
    image = np.load(filepath)
    axis_values = np.load(axis_value_file)

    pixels = image.reshape(-1, image.shape[-1])

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

    results = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(fit_spectrum, pixels), total=pixels.shape[0]))

    param_maps = np.array(results).reshape(image.shape[0], image.shape[1], 4)

    np.save(output_path + '_m_centre.npy', param_maps[:, :, 0])
    np.save(output_path + '_m_fwhm.npy', param_maps[:, :, 1])
    np.save(output_path + '_m_intensity.npy', param_maps[:, :, 2])

    with open(output_log, 'a') as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"CL fit script finished at {current_time}\n")
