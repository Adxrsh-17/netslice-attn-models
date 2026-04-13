#!/usr/bin/env python
# Auto-converted from urllc-slice.ipynb

# ---- markdown cell 1 ----
# # URLLC Slice — 5G Traffic Forecasting
# ## Hybrid VAR+GRU Deep Learning Comparative Study
# **Models:** VAR+GRU+N-BEATS | VAR+GRU+PatchTST | VAR+GRU+TimeMixer | VAR+GRU+TFT  
# **Slice:** Ultra-Reliable Low-Latency Communications (URLLC)  
# **Course:** 22AIE463 — Time Series Analysis | Group B5

# ---- code cell 2 ----
# ==============================================================================
# SETUP & CONFIGURATION
# ==============================================================================
import os, glob, json, warnings, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(42); tf.random.set_seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# === GPU STRATEGY ===
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Running on {strategy.num_replicas_in_sync} GPU(s)")
except:
    strategy = tf.distribute.get_strategy()
    print("Single Device Mode")

# === SLICE CONFIGURATION ===
SLICE_KEY   = "urllc"
SLICE_NAME  = "Youtube"
SLICE_LABEL = "URLLC"

CONFIG = {
    "window": 60, "forecast_horizon": 1, "epochs": 150,
    "batch_size": 256, "lr": 0.0001, "var_lags": 3,
    "target_slice": SLICE_NAME,
    "patch_size": 12, "patch_stride": 6
}

SLICE_MAP = {"Naver": "eMBB", "Youtube": "URLLC", "MMTC": "mMTC"}
FEATURE_MAP = {
    "throughput": "Throughput_bps", "packets": "Total_Packets",
    "jitter": "Jitter", "latency": "Avg_IAT",
    "reliability": "Retransmission_Ratio", "congestion": "Avg_Win_Size",
    "complexity": "Entropy_Score"
}
TARGET_FEATURES = list(FEATURE_MAP.keys())

# === OUTPUT DIRECTORIES ===
OUT = f"/kaggle/working/{SLICE_KEY}"
os.makedirs(f"{OUT}/models", exist_ok=True)
os.makedirs(f"{OUT}/plots", exist_ok=True)
os.makedirs(f"{OUT}/metrics", exist_ok=True)
print(f"Output directory: {OUT}")
print(f"Slice: {SLICE_LABEL} ({SLICE_KEY})")

# ---- code cell 3 ----
# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================
def load_and_prep_data(data_path, slice_name):
    print(f"Scanning: {data_path}")
    files = glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError("No dataset files found!")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
            dfs.append(df)
        except: pass

    raw_data = pd.concat(dfs, ignore_index=True)
    target_label = SLICE_MAP.get(slice_name, slice_name)
    print(f"Filtering for: '{target_label}'")

    if 'Slice_Type' not in raw_data.columns:
        if 'slice_label' in raw_data.columns:
            raw_data.rename(columns={'slice_label': 'Slice_Type'}, inplace=True)
        else:
            raw_data['Slice_Type'] = target_label

    df_slice = raw_data[raw_data['Slice_Type'].astype(str) == target_label].copy()
    if len(df_slice) == 0:
        raise ValueError(f"No data for '{target_label}'.")
    if 'Serial_No' in df_slice.columns:
        df_slice = df_slice.sort_values('Serial_No')

    print(f"Loaded {len(df_slice)} samples for {target_label}")
    print(f"Available columns: {list(df_slice.columns)}")

    final_df = pd.DataFrame()
    for target, source in FEATURE_MAP.items():
        final_df[target] = df_slice[source].copy() if source in df_slice.columns else 0.0
    return final_df.ffill().bfill().fillna(0)

# --- Execute ---
paths = ["/kaggle/input", ".", "/opt/spark/work-dir"]
data_path = next((p for p in paths if os.path.exists(p)), ".")
df = load_and_prep_data(data_path, CONFIG['target_slice'])

# --- Splits ---
n = len(df)
train_df = df.iloc[:int(0.7*n)]
val_df   = df.iloc[int(0.7*n):int(0.85*n)]
test_df  = df.iloc[int(0.85*n):]
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# --- Scaling ---
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled   = scaler.transform(val_df)
test_scaled  = scaler.transform(test_df)

# Save scaler
with open(f"{OUT}/metrics/{SLICE_KEY}_scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

# Quick summary
print("\n--- Data Summary ---")
print(df.describe().round(4))

# ---- markdown cell 4 ----
# ## 1. Pre-Modeling: Exploratory Data Plots & Statistical Tests

# ---- code cell 5 ----
# ==============================================================================
# 1. PRE-MODELING EDA
# ==============================================================================
raw_values = df.values

# --- [1A] Time Series Line Plot (All Features) ---
fig, axes = plt.subplots(len(TARGET_FEATURES), 1, figsize=(16, 3*len(TARGET_FEATURES)), sharex=True)
for i, feat in enumerate(TARGET_FEATURES):
    axes[i].plot(df[feat].values, linewidth=0.7, color=sns.color_palette()[i % 10])
    axes[i].set_ylabel(feat, fontsize=10)
    axes[i].set_title(f'{feat.capitalize()} — {SLICE_LABEL} Raw Time Series', fontsize=11, fontweight='bold')
plt.xlabel('Time Steps')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/01_timeseries_raw.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [1B] STL Decomposition ---
for idx, feat in enumerate(TARGET_FEATURES[:3]):
    series = df[feat].values
    if np.std(series) < 1e-10:
        print(f"Skipping STL for {feat} (flat)")
        continue
    try:
        decomp = seasonal_decompose(series[-min(2000, len(series)):], model='additive', period=CONFIG['window'])
        fig, (a1,a2,a3,a4) = plt.subplots(4,1, figsize=(14,8), sharex=True)
        a1.plot(decomp.observed, color='black'); a1.set_title(f'Observed — {feat}', fontweight='bold')
        a2.plot(decomp.trend, color='orange'); a2.set_title('Trend')
        a3.plot(decomp.seasonal, color='green'); a3.set_title('Seasonal')
        a4.plot(decomp.resid, color='red'); a4.set_title('Residual')
        plt.tight_layout()
        plt.savefig(f"{OUT}/plots/02_stl_{feat}.png", dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"STL failed for {feat}: {e}")

# --- [1C] ADF & KPSS Stationarity Tests ---
stationarity_results = {}
print(f"\n{'='*75}")
print(f"{'FEATURE':<15} | {'ADF Stat':<12} | {'ADF p-val':<12} | {'KPSS Stat':<12} | {'KPSS p-val':<12} | {'Verdict'}")
print(f"{'='*75}")
for feat in TARGET_FEATURES:
    series = df[feat].values
    if np.std(series) < 1e-10:
        stationarity_results[feat] = {"adf_stat": None, "adf_pval": None, "kpss_stat": None, "kpss_pval": None, "verdict": "Flat"}
        print(f"{feat:<15} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | Flat (zero variance)")
        continue
    adf = adfuller(series)
    kp = kpss(series, regression='c', nlags='auto')
    adf_ok = adf[1] < 0.05
    kpss_ok = kp[1] > 0.05
    verdict = "Stationary" if (adf_ok and kpss_ok) else ("Non-Stationary" if (not adf_ok and not kpss_ok) else "Mixed")
    stationarity_results[feat] = {"adf_stat": float(adf[0]), "adf_pval": float(adf[1]), "kpss_stat": float(kp[0]), "kpss_pval": float(kp[1]), "verdict": verdict}
    print(f"{feat:<15} | {adf[0]:<12.4f} | {adf[1]:<12.6f} | {kp[0]:<12.4f} | {kp[1]:<12.6f} | {verdict}")

with open(f"{OUT}/metrics/{SLICE_KEY}_stationarity.json", 'w') as f:
    json.dump(stationarity_results, f, indent=2)

# --- [1D] Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title(f'Feature Correlation Heatmap — {SLICE_LABEL}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/03_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [1E] Distribution Histograms ---
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for i, feat in enumerate(TARGET_FEATURES):
    axes[i].hist(df[feat].values, bins=50, color=sns.color_palette()[i], alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{feat}', fontweight='bold')
    axes[i].axvline(df[feat].mean(), color='red', linestyle='--', label='Mean')
    axes[i].axvline(df[feat].median(), color='blue', linestyle='--', label='Median')
    axes[i].legend(fontsize=8)
if len(TARGET_FEATURES) < len(axes):
    for j in range(len(TARGET_FEATURES), len(axes)):
        axes[j].set_visible(False)
plt.suptitle(f'Feature Distributions — {SLICE_LABEL}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/04_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [1F] Rolling Mean & Std ---
window_roll = 100
fig, axes = plt.subplots(len(TARGET_FEATURES), 1, figsize=(16, 3*len(TARGET_FEATURES)), sharex=True)
for i, feat in enumerate(TARGET_FEATURES):
    s = df[feat]
    axes[i].plot(s.values, alpha=0.3, color='gray', linewidth=0.5, label='Raw')
    axes[i].plot(s.rolling(window_roll).mean().values, color='blue', linewidth=1.5, label=f'Rolling Mean ({window_roll})')
    axes[i].plot(s.rolling(window_roll).std().values, color='red', linewidth=1.5, label=f'Rolling Std ({window_roll})')
    axes[i].set_ylabel(feat)
    axes[i].legend(fontsize=8, loc='upper right')
plt.suptitle(f'Rolling Statistics — {SLICE_LABEL}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/05_rolling_statistics.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [1G] Box Plots ---
fig, axes = plt.subplots(1, len(TARGET_FEATURES), figsize=(20, 5))
for i, feat in enumerate(TARGET_FEATURES):
    axes[i].boxplot(df[feat].dropna().values, patch_artist=True,
                    boxprops=dict(facecolor=sns.color_palette()[i], alpha=0.6))
    axes[i].set_title(feat, fontsize=10, fontweight='bold')
plt.suptitle(f'Box Plots — {SLICE_LABEL}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/06_boxplots.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [1H] Lag Plot ---
for feat in TARGET_FEATURES[:2]:
    s = df[feat].values
    if np.std(s) < 1e-10: continue
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, lag in enumerate([1, 7, 30]):
        if lag < len(s):
            axes[j].scatter(s[:-lag], s[lag:], alpha=0.2, s=2, color=sns.color_palette()[j])
            axes[j].set_title(f'Lag={lag}', fontweight='bold')
            axes[j].set_xlabel(f'{feat}(t)'); axes[j].set_ylabel(f'{feat}(t+{lag})')
    plt.suptitle(f'Lag Scatter Plot — {feat} ({SLICE_LABEL})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/07_lag_plot_{feat}.png", dpi=150, bbox_inches='tight')
    plt.show()

# --- [1I] Q-Q Plot ---
for feat in TARGET_FEATURES[:2]:
    s = df[feat].values
    if np.std(s) < 1e-10: continue
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(s, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot — {feat} ({SLICE_LABEL})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/08_qq_plot_{feat}.png", dpi=150, bbox_inches='tight')
    plt.show()

# --- [1J] Violin Plots ---
fig, ax = plt.subplots(figsize=(14, 6))
df_melted = df.melt(var_name='Feature', value_name='Value')
sns.violinplot(x='Feature', y='Value', data=df_melted, ax=ax, inner='quartile', scale='width')
ax.set_title(f'Violin Plots — {SLICE_LABEL}', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/plots/09_violin_plots.png", dpi=150, bbox_inches='tight')
plt.show()

print("Pre-Modeling EDA Complete")

# ---- markdown cell 6 ----
# ## 2. Baseline Model Identification (VAR Stage)

# ---- code cell 7 ----
# ==============================================================================
# 2. VAR BASELINE — ACF/PACF, AIC/BIC, Residual Extraction
# ==============================================================================

# --- [2A] ACF & PACF Plots (all features) ---
for feat_idx, feat in enumerate(TARGET_FEATURES):
    if np.std(df[feat].values) < 1e-10:
        print(f"Skipping ACF/PACF for {feat} (flat)")
        continue
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df[feat].values[-min(1500, len(df)):], lags=40, ax=axes[0], color='darkblue')
    axes[0].set_title(f'ACF — {feat}', fontweight='bold')
    plot_pacf(df[feat].values[-min(1500, len(df)):], lags=40, ax=axes[1], color='darkred', method='ywm')
    axes[1].set_title(f'PACF — {feat}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/10_acf_pacf_{feat}.png", dpi=150, bbox_inches='tight')
    plt.show()

# --- [2B] VAR Lag Selection ---
print("\nVAR Lag Selection:")
var_results = {}
try:
    var_temp = VAR(train_scaled)
    lag_order = var_temp.select_order(maxlags=15)
    print(lag_order.summary())
except Exception as e:
    print(f"Lag selection table failed: {e}")

# --- [2C] Fit VAR ---
print(f"\nFitting VAR(p={CONFIG['var_lags']})...")
var_model = VAR(train_scaled).fit(maxlags=CONFIG['var_lags'])
lag = var_model.k_ar

try:
    aic_val = float(var_model.aic)
    bic_val = float(var_model.bic)
    print(f"  AIC: {aic_val:.4f}")
    print(f"  BIC: {bic_val:.4f}")
    var_results['aic'] = aic_val
    var_results['bic'] = bic_val
except:
    print("  AIC/BIC: N/A (singular covariance)")
    var_results['aic'] = None
    var_results['bic'] = None
var_results['lags'] = int(lag)

# --- [2D] VAR Coefficient Heatmap ---
try:
    coef_matrix = var_model.coefs[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(coef_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=TARGET_FEATURES, yticklabels=TARGET_FEATURES, ax=ax)
    ax.set_title(f'VAR(1) Coefficient Matrix — {SLICE_LABEL}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/11_var_coefficients.png", dpi=150, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"VAR coefficient plot skipped: {e}")

# --- [2E] Granger Causality ---
try:
    print("\nGranger Causality (p-values at lag 3):")
    gc_matrix = np.zeros((len(TARGET_FEATURES), len(TARGET_FEATURES)))
    for i, f1 in enumerate(TARGET_FEATURES):
        for j, f2 in enumerate(TARGET_FEATURES):
            if i != j and np.std(df[f1].values) > 1e-10 and np.std(df[f2].values) > 1e-10:
                try:
                    test_data = df[[f2, f1]].values[-2000:]
                    gc = grangercausalitytests(test_data, maxlag=3, verbose=False)
                    gc_matrix[i, j] = gc[3][0]['ssr_ftest'][1]
                except:
                    gc_matrix[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(gc_matrix, annot=True, fmt='.3f', cmap='YlOrRd_r',
                xticklabels=TARGET_FEATURES, yticklabels=TARGET_FEATURES, ax=ax)
    ax.set_title(f'Granger Causality p-values — {SLICE_LABEL}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/12_granger_causality.png", dpi=150, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Granger causality failed: {e}")

# --- [2F] Extract VAR Residuals ---
def get_residuals(data, prev_data):
    if len(prev_data) < lag:
        return data, np.zeros_like(data)
    hist = np.vstack([prev_data[-lag:], data])
    pred = [var_model.forecast(hist[i-lag:i], 1)[0] for i in range(lag, len(hist))]
    return data - np.array(pred), np.array(pred)

res_train, _           = get_residuals(train_scaled[lag:], train_scaled[:lag])
res_val, _             = get_residuals(val_scaled, train_scaled)
res_test, var_pred_test = get_residuals(test_scaled, val_scaled)
print(f"\nResiduals — Train: {res_train.shape}, Val: {res_val.shape}, Test: {res_test.shape}")

# --- Sequence Generation ---
def make_seq(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_train, y_train = make_seq(res_train, CONFIG['window'])
X_val, y_val     = make_seq(res_val, CONFIG['window'])
X_test, y_test   = make_seq(res_test, CONFIG['window'])
print(f"Sequences — X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

with open(f"{OUT}/metrics/{SLICE_KEY}_var_info.json", 'w') as f:
    json.dump(var_results, f, indent=2)

# ---- markdown cell 8 ----
# ## 3. Model Architectures

# ---- code cell 9 ----
# ==============================================================================
# 3. MODEL ARCHITECTURE DEFINITIONS
# ==============================================================================
NUM_FEATURES = len(TARGET_FEATURES)

# ---- N-BEATS ----
class NBeatsBlock(layers.Layer):
    def __init__(self, units, backcast_length, forecast_length, **kwargs):
        super().__init__(**kwargs)
        self.units=units; self.backcast_length=backcast_length; self.forecast_length=forecast_length
        self.fc1=layers.Dense(units,activation='relu'); self.fc2=layers.Dense(units,activation='relu')
        self.fc3=layers.Dense(units,activation='relu'); self.fc4=layers.Dense(units,activation='relu')
        self.backcast_dense=layers.Dense(backcast_length); self.forecast_dense=layers.Dense(forecast_length)
    def call(self,x):
        h=self.fc4(self.fc3(self.fc2(self.fc1(x))))
        return self.backcast_dense(h), self.forecast_dense(h)

def build_nbeats(input_shape):
    inp=layers.Input(shape=input_shape)
    enc=layers.GRU(128,return_sequences=False,dropout=0.2)(inp)
    b1,f1=NBeatsBlock(64,128,input_shape[-1])(enc); r1=layers.Subtract()([enc,b1])
    b2,f2=NBeatsBlock(64,128,input_shape[-1])(r1);  r2=layers.Subtract()([r1,b2])
    b3,f3=NBeatsBlock(64,128,input_shape[-1])(r2)
    out=layers.Add()([f1,f2,f3])
    m=models.Model(inp,out,name="VAR_GRU_NBEATS")
    m.compile(optimizer=optimizers.Adam(CONFIG['lr']),loss=losses.Huber(delta=1.0),metrics=['mae'])
    return m

# ---- PatchTST ----
class PatchTransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha=layers.MultiHeadAttention(num_heads=num_heads,key_dim=d_model)
        self.ffn=models.Sequential([layers.Dense(ff_dim,activation='gelu'),layers.Dense(d_model)])
        self.ln1=layers.LayerNormalization(epsilon=1e-6); self.ln2=layers.LayerNormalization(epsilon=1e-6)
        self.d1=layers.Dropout(dropout); self.d2=layers.Dropout(dropout)
    def call(self,inputs,training=False):
        a=self.d1(self.mha(inputs,inputs),training=training); o1=self.ln1(inputs+a)
        return self.ln2(o1+self.d2(self.ffn(o1),training=training))

def build_patchtst(input_shape):
    inp=layers.Input(shape=input_shape)
    x=layers.GRU(64,return_sequences=True,dropout=0.1)(inp)
    d_model=64
    x=layers.Conv1D(d_model,kernel_size=CONFIG['patch_size'],strides=CONFIG['patch_stride'],padding='valid')(x)
    num_patches=(CONFIG['window']-CONFIG['patch_size'])//CONFIG['patch_stride']+1
    pos=tf.range(start=0,limit=num_patches,delta=1)
    x=x+layers.Embedding(input_dim=num_patches,output_dim=d_model)(pos)
    x=PatchTransformerBlock(d_model,4,128)(x)
    x=PatchTransformerBlock(d_model,4,128)(x)
    x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(64,activation='gelu')(x); x=layers.Dropout(0.1)(x)
    out=layers.Dense(input_shape[-1])(x)
    m=models.Model(inp,out,name="VAR_GRU_PatchTST")
    m.compile(optimizer=optimizers.Adam(CONFIG['lr']),loss=losses.Huber(delta=1.0),metrics=['mae'])
    return m

# ---- TimeMixer ----
class MixingBlock(layers.Layer):
    def __init__(self, seq_len, num_features, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.seq_len=seq_len; self.num_features=num_features
        self.norm1=layers.LayerNormalization(epsilon=1e-6); self.norm2=layers.LayerNormalization(epsilon=1e-6)
        self.time_dense=layers.Dense(seq_len,activation='gelu'); self.time_drop=layers.Dropout(dropout)
        self.feat1=layers.Dense(num_features*2,activation='gelu'); self.feat2=layers.Dense(num_features)
        self.feat_drop=layers.Dropout(dropout)
    def call(self,inputs,training=False):
        x=self.norm1(inputs); x=tf.transpose(x,perm=[0,2,1]); x=self.time_dense(x)
        x=self.time_drop(x,training=training); x=tf.transpose(x,perm=[0,2,1]); res=x+inputs
        x=self.norm2(res); x=self.feat1(x); x=self.feat_drop(x,training=training); x=self.feat2(x)
        return x+res

class MultiScaleTimeMixer(layers.Layer):
    def __init__(self, original_len, num_features, **kwargs):
        super().__init__(**kwargs)
        self.ol=original_len; self.nf=num_features
        self.m1=MixingBlock(original_len,num_features)
        self.p2=layers.AveragePooling1D(2,2,'valid'); self.m2=MixingBlock(original_len//2,num_features)
        self.p4=layers.AveragePooling1D(4,4,'valid'); self.m3=MixingBlock(original_len//4,num_features)
    def call(self,x):
        o1=self.m1(x)
        o2=self.m2(self.p2(x))
        o2=tf.squeeze(tf.image.resize(tf.expand_dims(o2,-1),[self.ol,self.nf],method='nearest'),-1)
        o3=self.m3(self.p4(x))
        o3=tf.squeeze(tf.image.resize(tf.expand_dims(o3,-1),[self.ol,self.nf],method='nearest'),-1)
        return o1+o2+o3

def build_timemixer(input_shape):
    inp=layers.Input(shape=input_shape)
    x=layers.GRU(64,return_sequences=True,dropout=0.2)(inp)
    x=layers.Dense(input_shape[-1])(x)
    x=MultiScaleTimeMixer(input_shape[0],input_shape[-1])(x)
    x=MultiScaleTimeMixer(input_shape[0],input_shape[-1])(x)
    x=layers.Flatten()(x); x=layers.Dense(64,activation='gelu')(x); x=layers.Dropout(0.1)(x)
    out=layers.Dense(input_shape[-1])(x)
    m=models.Model(inp,out,name="VAR_GRU_TimeMixer")
    m.compile(optimizer=optimizers.Adam(CONFIG['lr']),loss=losses.Huber(delta=1.0),metrics=['mae'])
    return m

# ---- TFT ----
class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout, **kwargs):
        super().__init__(**kwargs)
        self.units=units
        self.elu_dense=layers.Dense(units,activation='elu'); self.linear_dense=layers.Dense(units)
        self.drop=layers.Dropout(dropout); self.gate=layers.Dense(units,activation='sigmoid')
        self.norm=layers.LayerNormalization(); self.skip_project=None
    def build(self,input_shape):
        if input_shape[-1]!=self.units: self.skip_project=layers.Dense(self.units)
        super().build(input_shape)
    def call(self,x):
        skip=self.skip_project(x) if self.skip_project else x
        v=self.elu_dense(x); v=self.drop(v); v=self.linear_dense(v); v=v*self.gate(x)
        return self.norm(skip+v)

def build_tft(input_shape):
    inp=layers.Input(shape=input_shape)
    x=GatedResidualNetwork(64,0.1)(inp)
    x=layers.GRU(128,return_sequences=True,dropout=0.2)(x)
    x=layers.GRU(64,return_sequences=True,dropout=0.2)(x)
    x=layers.MultiHeadAttention(num_heads=4,key_dim=32)(x,x)
    x=layers.LayerNormalization()(x)
    x=layers.GlobalAveragePooling1D()(x)
    x=GatedResidualNetwork(32,0.1)(x)
    out=layers.Dense(input_shape[-1])(x)
    m=models.Model(inp,out,name="VAR_GRU_TFT")
    m.compile(optimizer=optimizers.Adam(CONFIG['lr']),loss=losses.Huber(delta=1.0),metrics=['mae'])
    return m

# Registry
MODEL_BUILDERS = {
    "nbeats":    build_nbeats,
    "patchtst":  build_patchtst,
    "timemixer": build_timemixer,
    "tft":       build_tft
}

MODEL_DISPLAY = {
    "nbeats": "VAR+GRU+N-BEATS", "patchtst": "VAR+GRU+PatchTST",
    "timemixer": "VAR+GRU+TimeMixer", "tft": "VAR+GRU+TFT"
}

print("All 4 model architectures defined.")
for k, builder in MODEL_BUILDERS.items():
    m = builder((CONFIG['window'], NUM_FEATURES))
    print(f"  {MODEL_DISPLAY[k]}: {m.count_params():,} parameters")
    del m

# ---- markdown cell 10 ----
# ## 4. Training Pipeline

# ---- code cell 11 ----
# ==============================================================================
# 4. TRAIN ALL 4 MODELS
# ==============================================================================
trained_models = {}
training_histories = {}

for model_key, builder in MODEL_BUILDERS.items():
    print(f"\n{'='*60}")
    print(f"  TRAINING: {MODEL_DISPLAY[model_key]}")
    print(f"{'='*60}")

    with strategy.scope():
        model = builder((CONFIG['window'], NUM_FEATURES))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'] * strategy.num_replicas_in_sync,
        callbacks=[
            callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_mae'),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
            callbacks.CSVLogger(f"{OUT}/metrics/{SLICE_KEY}_{model_key}_history.csv")
        ],
        verbose=1
    )

    trained_models[model_key] = model
    training_histories[model_key] = history.history

    # Save model
    model.save(f"{OUT}/models/{SLICE_KEY}_{model_key}.h5")
    print(f"  Saved: {OUT}/models/{SLICE_KEY}_{model_key}.h5")

    # --- Training vs Validation Loss Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history.history['loss'], label='Train Loss (Huber)', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss (Huber)', color='red')
    ax1.set_title(f'Loss — {MODEL_DISPLAY[model_key]}', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Huber Loss'); ax1.legend()

    ax2.plot(history.history['mae'], label='Train MAE', color='blue')
    ax2.plot(history.history['val_mae'], label='Val MAE', color='red')
    ax2.set_title(f'MAE — {MODEL_DISPLAY[model_key]}', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE'); ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/13_training_loss_{model_key}.png", dpi=150, bbox_inches='tight')
    plt.show()

print("\nAll 4 models trained and saved!")

# ---- markdown cell 12 ----
# ## 5. Comprehensive Evaluation & Diagnostics

# ---- code cell 13 ----
# ==============================================================================
# 5. COMPREHENSIVE EVALUATION
# ==============================================================================
all_metrics = {}
all_predictions = {}

y_true_real = scaler.inverse_transform(test_scaled[CONFIG['window']:CONFIG['window']+len(X_test)])

for model_key, model in trained_models.items():
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {MODEL_DISPLAY[model_key]}")
    print(f"{'='*60}")

    resid_pred = model.predict(X_test, batch_size=CONFIG['batch_size'])
    L = len(resid_pred)
    final_pred = var_pred_test[CONFIG['window']:CONFIG['window']+L] + resid_pred
    y_pred_real = np.maximum(scaler.inverse_transform(final_pred), 0)
    all_predictions[model_key] = y_pred_real

    # --- Per-Feature Metrics ---
    model_metrics = {"features": {}, "overall": {}}
    print(f"\n{'FEATURE':<15} | {'RMSE':<12} | {'MAE':<12} | {'R²':<12}")
    print("-" * 55)
    rmses, maes, r2s = [], [], []
    for i, feat in enumerate(TARGET_FEATURES):
        rmse = np.sqrt(mean_squared_error(y_true_real[:L, i], y_pred_real[:, i]))
        mae = mean_absolute_error(y_true_real[:L, i], y_pred_real[:, i])
        r2 = r2_score(y_true_real[:L, i], y_pred_real[:, i])
        rmses.append(rmse); maes.append(mae); r2s.append(r2)
        model_metrics["features"][feat] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        print(f"{feat:<15} | {rmse:<12.4f} | {mae:<12.4f} | {r2:<12.4f}")

    model_metrics["overall"] = {
        "avg_rmse": float(np.mean(rmses)), "avg_mae": float(np.mean(maes)), "avg_r2": float(np.mean(r2s))
    }
    all_metrics[model_key] = model_metrics

    # --- [5A] Forecast vs Actual (All Features Grid) ---
    fig, axes = plt.subplots(len(TARGET_FEATURES), 1, figsize=(16, 3*len(TARGET_FEATURES)), sharex=True)
    ZOOM = min(300, L)
    for i, feat in enumerate(TARGET_FEATURES):
        axes[i].plot(y_true_real[-ZOOM:, i], color='black', alpha=0.8, linewidth=1, label='Actual')
        axes[i].plot(y_pred_real[-ZOOM:, i], color='red', linestyle='--', linewidth=1.5, label='Predicted')
        axes[i].set_ylabel(feat); axes[i].legend(fontsize=8, loc='upper right')
    plt.suptitle(f'Forecast vs Actual (All Features) — {MODEL_DISPLAY[model_key]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/14_forecast_all_{model_key}.png", dpi=150, bbox_inches='tight')
    plt.show()

    # --- [5B] Zoomed Forecast (Throughput) ---
    plt.figure(figsize=(14, 5))
    plt.plot(y_true_real[-ZOOM:, 0], color='black', alpha=0.8, linewidth=1.5, label='Actual (Throughput)')
    plt.plot(y_pred_real[-ZOOM:, 0], color='red', linestyle='--', linewidth=2, label=f'{MODEL_DISPLAY[model_key]} Forecast')
    r2_thr = r2_score(y_true_real[:L, 0], y_pred_real[:, 0])
    plt.title(f'Throughput Forecast — {MODEL_DISPLAY[model_key]} (R²={r2_thr:.4f})', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps'); plt.ylabel('Throughput (bps)'); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/15_forecast_throughput_{model_key}.png", dpi=150, bbox_inches='tight')
    plt.show()

    # --- [5C] Residual Diagnostics ---
    for feat_idx in [0, 3]:
        feat = TARGET_FEATURES[feat_idx]
        residuals = y_true_real[:L, feat_idx] - y_pred_real[:, feat_idx]
        if np.std(residuals) < 1e-10:
            print(f"  Skipping residual tests for {feat} (zero variance)")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        plot_acf(residuals, lags=40, ax=axes[0,0], color='purple')
        axes[0,0].set_title(f'Residual ACF — {feat}', fontweight='bold')
        axes[0,1].plot(residuals**2, color='gray', alpha=0.7)
        axes[0,1].set_title(f'Squared Residuals — {feat}', fontweight='bold')
        axes[1,0].hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black', density=True)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1,0].plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()), 'r-', linewidth=2)
        axes[1,0].set_title(f'Residual Distribution — {feat}', fontweight='bold')
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title(f'Q-Q Plot (Residuals) — {feat}', fontweight='bold')
        plt.suptitle(f'{MODEL_DISPLAY[model_key]} — Residual Diagnostics ({feat})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{OUT}/plots/16_residuals_{model_key}_{feat}.png", dpi=150, bbox_inches='tight')
        plt.show()

        try:
            lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
            lb_p = lb['lb_pvalue'].values[0]
            arch = het_arch(residuals)
            model_metrics["features"][feat]["ljung_box_pval"] = float(lb_p)
            model_metrics["features"][feat]["arch_lm_pval"] = float(arch[1])
            print(f"  {feat} — Ljung-Box p={lb_p:.6f} ({'White Noise' if lb_p>0.05 else 'Structure'}) | ARCH LM p={arch[1]:.6f} ({'No Clustering' if arch[1]>0.05 else 'Clustering'})")
        except Exception as e:
            print(f"  Residual tests failed for {feat}: {e}")

# --- [5D] Model Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model_names_disp = [MODEL_DISPLAY[k] for k in all_metrics.keys()]
avg_rmses = [all_metrics[k]["overall"]["avg_rmse"] for k in all_metrics.keys()]
avg_r2s = [all_metrics[k]["overall"]["avg_r2"] for k in all_metrics.keys()]
colors_bar = ['#8B0000', '#2E7D32', '#E65100', '#9C27B0']

axes[0].barh(model_names_disp, avg_rmses, color=colors_bar[:len(model_names_disp)])
axes[0].set_title('Average RMSE (lower is better)', fontweight='bold')
axes[0].set_xlabel('RMSE')
axes[1].barh(model_names_disp, avg_r2s, color=colors_bar[:len(model_names_disp)])
axes[1].set_title('Average R² (higher is better)', fontweight='bold')
axes[1].set_xlabel('R²')
plt.suptitle(f'Model Comparison — {SLICE_LABEL}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/17_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [5E] Radar Chart ---
best_key = max(all_metrics.keys(), key=lambda k: all_metrics[k]["overall"]["avg_r2"])
best_r2_per_feat = [max(0, all_metrics[best_key]["features"][f]["r2"]) * 100 for f in TARGET_FEATURES]
angles = np.linspace(0, 2*np.pi, len(TARGET_FEATURES), endpoint=False).tolist()
angles += angles[:1]; best_r2_per_feat += best_r2_per_feat[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, best_r2_per_feat, color='crimson', alpha=0.25)
ax.plot(angles, best_r2_per_feat, 'o-', color='crimson', linewidth=2)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(TARGET_FEATURES)
ax.set_ylim(0, 100); ax.set_title(f'Per-KPI R² (%) — {MODEL_DISPLAY[best_key]} ({SLICE_LABEL})', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUT}/plots/18_radar_r2.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [5F] All Models Overlay ---
plt.figure(figsize=(16, 6)); ZOOM = min(200, L)
plt.plot(y_true_real[-ZOOM:, 0], color='black', linewidth=2, label='Actual', alpha=0.9)
line_styles = ['-', '--', '-.', ':']
for idx, (mk, yp) in enumerate(all_predictions.items()):
    plt.plot(yp[-ZOOM:, 0], color=colors_bar[idx], linestyle=line_styles[idx], linewidth=1.5, label=MODEL_DISPLAY[mk], alpha=0.8)
plt.title(f'All Models vs Actual — Throughput ({SLICE_LABEL})', fontsize=14, fontweight='bold')
plt.xlabel('Time Steps'); plt.ylabel('Throughput (bps)'); plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/plots/19_all_models_overlay.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [5G] Feature-wise RMSE ---
fig, ax = plt.subplots(figsize=(14, 6)); x = np.arange(len(TARGET_FEATURES)); width = 0.2
for idx, (mk, met) in enumerate(all_metrics.items()):
    rmse_vals = [met["features"][f]["rmse"] for f in TARGET_FEATURES]
    ax.bar(x + idx*width, rmse_vals, width, label=MODEL_DISPLAY[mk], color=colors_bar[idx])
ax.set_xticks(x + width*1.5); ax.set_xticklabels(TARGET_FEATURES, rotation=45)
ax.set_title(f'Per-Feature RMSE Comparison — {SLICE_LABEL}', fontweight='bold')
ax.set_ylabel('RMSE'); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/plots/20_feature_rmse_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [5H] Error Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10)); axes = axes.flatten()
for idx, (mk, yp) in enumerate(all_predictions.items()):
    errors = y_true_real[:L, 0] - yp[:, 0]
    axes[idx].hist(errors, bins=50, color=colors_bar[idx], alpha=0.7, edgecolor='black', density=True)
    axes[idx].axvline(0, color='black', linestyle='--')
    axes[idx].set_title(f'{MODEL_DISPLAY[mk]} — Error Distribution', fontweight='bold')
plt.suptitle(f'Prediction Error Distributions — {SLICE_LABEL}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/21_error_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [5I] CUSUM ---
plt.figure(figsize=(14, 4))
best_errors = y_true_real[:L, 0] - all_predictions[best_key][:, 0]
cusum = np.cumsum(best_errors - best_errors.mean())
plt.plot(cusum, color='darkblue', linewidth=1.5); plt.axhline(0, color='red', linestyle='--')
plt.title(f'CUSUM Chart — {MODEL_DISPLAY[best_key]} ({SLICE_LABEL})', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/22_cusum.png", dpi=150, bbox_inches='tight')
plt.show()

# --- [5J] Spectral Density ---
from numpy.fft import fft, fftfreq
s = y_true_real[:L, 0]
if np.std(s) > 1e-10:
    N = len(s); yf = 2.0/N * np.abs(fft(s)[:N//2]); xf = fftfreq(N, 1)[:N//2]
    plt.figure(figsize=(14, 4)); plt.plot(xf[1:], yf[1:], color='navy', linewidth=0.8)
    plt.title(f'Power Spectral Density — throughput ({SLICE_LABEL})', fontweight='bold')
    plt.xlabel('Frequency'); plt.ylabel('Amplitude'); plt.tight_layout()
    plt.savefig(f"{OUT}/plots/23_spectral_density.png", dpi=150, bbox_inches='tight'); plt.show()

print("\nEvaluation complete!")

# ---- code cell 14 ----
# ==============================================================================
# 6. SAVE ALL ARTIFACTS
# ==============================================================================
webapp_metrics = {"slice": SLICE_KEY, "slice_label": SLICE_LABEL, "models": {}}

for mk in all_metrics:
    mm = all_metrics[mk]
    webapp_metrics["models"][mk] = {
        "display_name": MODEL_DISPLAY[mk],
        "throughput_rmse": mm["features"]["throughput"]["rmse"],
        "latency_rmse": mm["features"]["latency"]["rmse"],
        "jitter_rmse": mm["features"]["jitter"]["rmse"],
        "packets_rmse": mm["features"]["packets"]["rmse"],
        "reliability_rmse": mm["features"]["reliability"]["rmse"],
        "congestion_rmse": mm["features"]["congestion"]["rmse"],
        "avg_rmse": mm["overall"]["avg_rmse"],
        "avg_mae": mm["overall"]["avg_mae"],
        "avg_r2": mm["overall"]["avg_r2"],
        "per_feature": mm["features"]
    }

best_key = max(all_metrics.keys(), key=lambda k: all_metrics[k]["overall"]["avg_r2"])
webapp_metrics["radar"] = {
    "best_model": MODEL_DISPLAY[best_key],
    "labels": TARGET_FEATURES,
    "values": [max(0, all_metrics[best_key]["features"][f]["r2"]) * 100 for f in TARGET_FEATURES]
}
webapp_metrics["charts"] = {
    "model_names": [MODEL_DISPLAY[mk] for mk in MODEL_BUILDERS.keys()],
    "rmse": [all_metrics[mk]["features"]["throughput"]["rmse"] for mk in MODEL_BUILDERS.keys()],
    "mae": [all_metrics[mk]["features"]["throughput"]["mae"] for mk in MODEL_BUILDERS.keys()]
}

best_pred = all_predictions[best_key]
EXPORT_LEN = min(300, len(best_pred))
webapp_metrics["predictions"] = {
    "model": MODEL_DISPLAY[best_key],
    "labels": list(range(EXPORT_LEN)),
    "actual": y_true_real[-EXPORT_LEN:, 0].tolist(),
    "predicted": best_pred[-EXPORT_LEN:, 0].tolist(),
    "r2_score": float(r2_score(y_true_real[:len(best_pred), 0], best_pred[:, 0]))
}

webapp_metrics["results_table"] = []
for mk in MODEL_BUILDERS.keys():
    mm = all_metrics[mk]
    webapp_metrics["results_table"].append({
        "model": MODEL_DISPLAY[mk],
        "throughputRMSE": round(mm["features"]["throughput"]["rmse"], 4),
        "latencyRMSE": round(mm["features"]["latency"]["rmse"], 6),
        "jitterRMSE": round(mm["features"]["jitter"]["rmse"], 6),
        "packetsRMSE": round(mm["features"]["packets"]["rmse"], 4),
        "r2": round(mm["overall"]["avg_r2"], 4),
        "bestSlice": SLICE_LABEL
    })

with open(f"{OUT}/metrics/{SLICE_KEY}_webapp_metrics.json", 'w') as f:
    json.dump(webapp_metrics, f, indent=2)

for mk, pred in all_predictions.items():
    np.save(f"{OUT}/metrics/{SLICE_KEY}_{mk}_predictions.npy", pred)
np.save(f"{OUT}/metrics/{SLICE_KEY}_true_values.npy", y_true_real)

print(f"\n{'='*60}")
print(f"  ALL ARTIFACTS SAVED TO: {OUT}")
print(f"{'='*60}")
for root, dirs, files_list in os.walk(OUT):
    level = root.replace(OUT, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files_list:
        fsize = os.path.getsize(os.path.join(root, file))
        print(f'{subindent}{file} ({fsize/1024:.1f} KB)')

print(f"\nDownload the entire '{SLICE_KEY}/' folder from Kaggle Output")
print("and place contents in your workspace as described in KAGGLE_INSTRUCTIONS.md")
