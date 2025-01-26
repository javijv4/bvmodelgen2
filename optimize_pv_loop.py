
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from shapely import Polygon

def compute_pv_area(x_coords, y_coords):
    polygon = Polygon(zip(x_coords, y_coords))
    area = polygon.area
    return area

def get_pressure_trace(pres_func, time, time_shift=0):
    return pres_func(time - time_shift)

def get_volume_trace(vol_func, time, time_shift=0):
    return vol_func(time - time_shift)

# Read volume and pressure data
lv_vol = ...
lv_pres = ...
vol_time = ...
pres_time = ...
vol_time_cycle = np.max(vol_time)
pres_time_cycle = np.max(pres_time)

# Extend data
lv_vol_ext = np.concatenate(lv_vol, lv_vol, lv_vol)
lv_pres_ext = np.concatenate(lv_pres, lv_pres, lv_pres)
vol_time_ext = np.concatenate(vol_time-vol_time_cycle, vol_time, vol_time + vol_time_cycle)
pres_time_ext = np.concatenate(pres_time-pres_time_cycle, pres_time, pres_time+pres_time_cycle)


# Plots extended vol, pres and time


# Interpolate volume and pressure data using normalized time
vol_norm_time = np.linspace(-1, 2, len(lv_vol_ext))
lv_vol_interp = interp1d(vol_norm_time, lv_vol_ext)
pres_norm_time = np.linspace(-1, 2, len(lv_pres_ext))
lv_pres_interp = interp1d(pres_norm_time, lv_pres_ext)

norm_time = np.linspace(0.,1.,200)

# Optimize PV area
def optimize_func(vol_time_shift):
    lv_vol_shift = get_volume_trace(lv_vol_interp, norm_time, vol_time_shift)
    lv_pres_shift = get_pressure_trace(lv_pres_interp, norm_time)
    lv_area = compute_pv_area(lv_vol_shift, lv_pres_shift)
    return -lv_area

x0 = 0.0
bounds = [(-1/3, 1/3)]
sol = minimize(optimize_func, x0, method='trust-constr', bounds=bounds)

# Evaluate the solution
lv_vol_shift = get_volume_trace(lv_vol_interp, vol_time, sol.x)
lv_pres_shift = get_pressure_trace(lv_pres_interp, pres_time)
time = norm_time*pres_time_cycle

# Plot the optimized PV loop