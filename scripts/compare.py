# %%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# %%
#Wave Gauge Files (All parameters equal except C)
wgC10_file = r"C:\Users\maple\Documents\Academia\UW\SPHyNX_Lab\Model_output\cx_p05_wec05_d2_wmd025_sg05_t30\c10_p05_wt05_d2_wmd03_t30\data\WaveGage.txt"
wgC50_file = r"C:\Users\maple\Documents\Academia\UW\SPHyNX_Lab\Model_output\cx_p05_wec05_d2_wmd025_sg05_t30\c50_p05_wt05_d2_wmd03_t30\data\WaveGage.txt"

#Object Forces Files (All parameters equal except C)
ofC10_file = r"C:\Users\maple\Documents\Academia\UW\SPHyNX_Lab\Model_output\cx_p05_wec05_d2_wmd025_sg05_t30\c10_p05_wt05_d2_wmd03_t30\data\objectforces.txt"
ofC50_file = r"C:\Users\maple\Documents\Academia\UW\SPHyNX_Lab\Model_output\cx_p05_wec05_d2_wmd025_sg05_t30\c50_p05_wt05_d2_wmd03_t30\data\objectforces.txt"

#OSWEC Data Files (All parameters equal except C)
rbC10_file = r"C:\Users\maple\Documents\Academia\UW\SPHyNX_Lab\Model_output\cx_p05_wec05_d2_wmd025_sg05_t30\c10_p05_wt05_d2_wmd03_t30\data\rbdata.txt"
rbC50_file = r"C:\Users\maple\Documents\Academia\UW\SPHyNX_Lab\Model_output\cx_p05_wec05_d2_wmd025_sg05_t30\c50_p05_wt05_d2_wmd03_t30\data\rbdata.txt"


# %%
#Dataframe sets
#Wave Gauges
wgC10_df = pd.read_csv(wgC10_file, sep=r"\s+");
wgC50_df = pd.read_csv(wgC50_file, sep=r"\s+");

#Object Forces
ofC10_df = pd.read_csv(ofC10_file, sep=r"\s+");
ofC50_df = pd.read_csv(ofC50_file, sep=r"\s+");

#OSWEC Data
rbC10_df = pd.read_csv(rbC10_file, sep=r"\s+");
rbC50_df = pd.read_csv(rbC50_file, sep=r"\s+");


# header meanings:
#CM*_X, CM*_Y, CM*_Z → Center of Mass position
#Q*_1, Q*_I, Q*_J, Q*_K → Orientation quaternion

# %%
# --- 1) Put all runs in one place ---
runs = {
    10:  {"wg": wgC10_df,   "rb": rbC10_df,   "of": ofC10_df},
    50:  {"wg": wgC50_df,   "rb": rbC50_df,   "of": ofC50_df},
#    100: {"wg": wgC100_df,  "rb": rbC100_df,  "of": ofC100_df},
#    500: {"wg": wgC500_df,  "rb": rbC500_df,  "of": ofC500_df},
#    1000:{"wg": wgC1000_df, "rb": rbC1000_df, "of": ofC1000_df},
}

# %%
# --- 2) Settings you can tweak ---
gauge_col = "zgage16"
tmin, tmax = 5, 30        # match the paper-style steady window (adjust as needed)
use_degrees = True            # plot theta in degrees
eps = 1e-12    #ASK WHAT THIS IS

def steady_window_mask(t, tmin, tmax):
    return (t >= tmin) & (t <= tmax)

def cycle_amplitude_from_signal(t, y, T_est=None):

   # Regular-wave amplitude estimate: median peak-to-trough / 2.
   # (replace this with scipy.signal.find_peaks later?)

    #  estimate: (95th - 5th)/2 in steady state
    return 0.5 * (np.nanpercentile(y, 95) - np.nanpercentile(y, 5))

def add_omega(df, theta_col="Q0_theta_rad", omega_col="Omega_rad_s"):
    t = df["time"].to_numpy()
    th = df[theta_col].to_numpy()
    # centered gradient (handles nonuniform dt too)
    df[omega_col] = np.gradient(th, t)
    return df

# %%
# n^ = unit vector along rotation axis
# θ = rotation angle
# w=cos(θ/2)
# (x,y,z)=n^sin(θ/2)

#R= [​cosθ 0 −sinθ] Quaternion rotation matrix
#   [​ 0   1   0  ]
#   [​sinθ 0  cosθ​]

# sinθ =     2(wy+xz)
# cosθ = 1 − 2(y2+z2)

def add_theta(df,
             w_col='Q0_1',
             x_col='Q0_I',
             y_col='Q0_J',
             z_col='Q0_K',
             rad_col='Q0_theta_rad',
             deg_col='Q0_theta_deg'):
#    Compute rotation angle theta from quaternion components
#    and add results to the DataFrame.
#
#    Parameters
#    ----------
#    df : pandas.DataFrame
#        DataFrame containing quaternion columns
#    w_col, x_col, y_col, z_col : str
#        Column names for quaternion components
#    rad_col, deg_col : str
#        Names for output angle columns
#
#    Returns
#    -------
#    pandas.DataFrame
#        DataFrame with added theta columns

    w = df[w_col]
    x = df[x_col]
    y = df[y_col]
    z = df[z_col]

    theta = np.arctan2(
        2 * (w*y + x*z),
        1 - 2 * (y*y + z*z)
    )

    df[rad_col] = theta
    df[deg_col] = np.degrees(theta)

    return df
# %%
MWL = 2.0
gauge_col = "zgage20"
T = 3/2
omega = 2*np.pi / T
tau_shift = 0.0   # seconds; set later if you want alignment

def make_eta_measured(wg, gauge_col, MWL):
    return wg[gauge_col].to_numpy() - MWL

def make_eta_theory(t, A_run, omega, tau=0.0):
    return A_run * np.sin(omega * (t - tau))

# Example for one run (C=10)
wg = runs[10]["wg_proc"]
t = wg["time"].to_numpy()
eta_meas = make_eta_measured(wg, gauge_col, MWL)

# compute amplitude from steady window
mwg = (t >= tmin) & (t <= tmax)
A_run = 0.5 * (np.nanpercentile(eta_meas[mwg], 95) - np.nanpercentile(eta_meas[mwg], 5))

eta_theory = make_eta_theory(t, A_run, omega, tau=tau_shift)

# %%
#Change to make: Use mean amplitude in constant wave function value (don't use data directly from forward wave gauge)

c_list = [10, 50, 100, 500, 1000]
fig, axes = plt.subplots(len(c_list), 1, figsize=(12, 14), sharex=True)

for ax, c in zip(axes, c_list):
    wg = runs[c]["wg_proc"]
    rb = runs[c]["rb_proc"]

    # steady window slices
    mwg = steady_window_mask(wg["time"].to_numpy(), tmin, tmax)
    mrb = steady_window_mask(rb["time"].to_numpy(), tmin, tmax)

    # wave gauge
    ax.plot(wg.loc[mwg, "time"], wg.loc[mwg, gauge_col], label="wave")
    ax.set_ylabel("wave (m)")
    ax.set_title(f"(c) c={c}")

    # response on twin axis
    axr = ax.twinx()
    resp_col = "Q0_theta_deg" if use_degrees else "Q0_theta_rad"
    axr.plot(rb.loc[mrb, "time"], rb.loc[mrb, resp_col], label="response")
    axr.set_ylabel("theta (deg)" if use_degrees else "theta (rad)")

# x label only on bottom
axes[-1].set_xlabel("time (s)")

plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(1, 3, figsize=(16, 4))

# 1) c vs theta_max
sc0 = axs[0].scatter(metrics["c"], metrics["theta_max_deg"], c=metrics["A_gauge_m"])
axs[0].set_xlabel("c")
axs[0].set_ylabel("max |theta| (deg)")
plt.colorbar(sc0, ax=axs[0], label="gauge amplitude A (m)")

# 2) c vs Omega_max
sc1 = axs[1].scatter(metrics["c"], metrics["Omega_max_rad_s"], c=metrics["A_gauge_m"])
axs[1].set_xlabel("c")
axs[1].set_ylabel("max |Omega| (rad/s)")
plt.colorbar(sc1, ax=axs[1], label="gauge amplitude A (m)")

# 3) c vs mean power
sc2 = axs[2].scatter(metrics["c"], metrics["P_mean"], c=metrics["A_gauge_m"])
axs[2].set_xlabel("c")
axs[2].set_ylabel("mean PTO power ~ c * mean(Omega^2)")
plt.colorbar(sc2, ax=axs[2], label="gauge amplitude A (m)")

for ax in axs:
    ax.set_xscale("log")  # optional but often helpful (10 -> 1000)
plt.tight_layout()
plt.show()

#%%
# UNTESTED: Code purely fro gpt. try out when you have appropriate data set.
#--- USER: pretend these are your depth-only runs (same c, same wave settings) ---
depth_runs = {
    1.0: {"wg_file": "/path/to/d1/.../WaveGage.txt",
          "rb_file": "/path/to/d1/.../rbdata.txt",
          "c": 100},
    2.0: {"wg_file": "/path/to/d2/.../WaveGage.txt",
          "rb_file": "/path/to/d2/.../rbdata.txt",
          "c": 100},
    3.0: {"wg_file": "/path/to/d3/.../WaveGage.txt",
          "rb_file": "/path/to/d3/.../rbdata.txt",
          "c": 100},
}

MWL = 2.0
gauge_col = "zgage16"
tmin, tmax = 150, 500

def steady_mask(t):
    return (t >= tmin) & (t <= tmax)

def amp_robust(y):
    # robust amplitude estimate in steady state
    return 0.5 * (np.nanpercentile(y, 95) - np.nanpercentile(y, 5))

def add_omega(df, theta_col="Q0_theta_rad", omega_col="Omega_rad_s"):
    t = df["time"].to_numpy()
    th = df[theta_col].to_numpy()
    df[omega_col] = np.gradient(th, t)
    return df

rows = []
for h, info in depth_runs.items():
    wg = pd.read_csv(info["wg_file"], sep=r"\s+")
    rb = pd.read_csv(info["rb_file"], sep=r"\s+")
    c  = info["c"]

    rb = add_theta(rb)
    rb = add_omega(rb)

    mwg = steady_mask(wg["time"].to_numpy())
    mrb = steady_mask(rb["time"].to_numpy())

    eta = (wg.loc[mwg, gauge_col].to_numpy() - MWL)
    A_h = amp_robust(eta)

    Omega = rb.loc[mrb, "Omega_rad_s"].to_numpy()
    P_mean = c * np.nanmean(Omega**2)

    rows.append({"h_m": h, "A_m": A_h, "P_mean": P_mean})

df_h = pd.DataFrame(rows).sort_values("h_m")

# --- dashed "expected linear" line ---
alpha, beta = np.polyfit(df_h["h_m"], df_h["P_mean"], 1)
h_line = np.linspace(df_h["h_m"].min(), df_h["h_m"].max(), 100)
P_line = alpha*h_line + beta

plt.figure(figsize=(7,4))
plt.scatter(df_h["h_m"], df_h["P_mean"], label="simulated")
plt.plot(h_line, P_line, linestyle="--", label=f"linear fit: P={alpha:.3g}h+{beta:.3g}")
plt.xlabel("water depth h (m)")
plt.ylabel("mean PTO power ~ c * mean(Omega^2)")
plt.legend()
plt.tight_layout()
plt.show()

df_h

#%%
# UNTESTED: Code purely fro gpt. try out when you have appropriate data set.
# pick reference run (middle depth is a nice default)
h0 = df_h["h_m"].median()
ref = df_h.iloc[(df_h["h_m"] - h0).abs().argmin()]
A0 = ref["A_m"]
P0 = ref["P_mean"]

df_h["P_expected_A2"] = P0 * (df_h["A_m"] / (A0 + 1e-12))**2

plt.figure(figsize=(7,4))
plt.scatter(df_h["h_m"], df_h["P_mean"], label="simulated")
plt.plot(df_h["h_m"], df_h["P_expected_A2"], linestyle="--",
         label=r"expected: $P \propto A(h)^2$ (calibrated)")
plt.xlabel("water depth h (m)")
plt.ylabel("mean PTO power ~ c * mean(Omega^2)")
plt.legend()
plt.tight_layout()