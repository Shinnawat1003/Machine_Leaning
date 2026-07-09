"""
Bias-Variance Decomposition Lab (Compact Version)
เปรียบเทียบ bias^2/variance ของ 3 โมเดล บน 2 ฟังก์ชันเป้าหมาย ด้วย simulation (Monte Carlo)
Models: Constant h=b | Linear h=w0+w1x | Linear-thru-origin h=wx | n=2 ตัวอย่าง/dataset
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGETS = {
    'sin(pi*x)': lambda x: np.sin(np.pi * x),
    'x^2': lambda x: x ** 2,
}

def fit_constant(X, y):
    return {'b': np.mean(y)}

def predict_constant(p, X):
    return np.full(np.shape(X), p['b'])

def fit_linear(X, y):
    Phi = np.column_stack([np.ones(len(X)), X])
    w0, w1 = np.linalg.lstsq(Phi, y, rcond=None)[0]
    return {'w0': w0, 'w1': w1}

def predict_linear(p, X):
    return p['w0'] + p['w1'] * np.asarray(X)

def fit_linear_origin(X, y):
    X = np.asarray(X).reshape(-1, 1)
    w = np.linalg.lstsq(X, y, rcond=None)[0][0]
    return {'w': w}

def predict_linear_origin(p, X):
    return p['w'] * np.asarray(X)

MODELS = {
    'Constant': (fit_constant, predict_constant),
    'Linear': (fit_linear, predict_linear),
    'Linear thru origin': (fit_linear_origin, predict_linear_origin),
}

def simulate_bias_variance(f, fit_fn, predict_fn, n_samples=2, n_datasets=50000, n_test=300):
    """Monte Carlo estimate ของ bias^2, variance, Eout จากชุดข้อมูลขนาด n_samples จำนวน n_datasets ชุด"""
    x_test = np.linspace(-1, 1, n_test)
    preds = np.zeros((n_datasets, n_test))

    for i in range(n_datasets):
        X = np.random.uniform(-1, 1, n_samples)
        y = f(X)
        params = fit_fn(X, y)
        preds[i] = predict_fn(params, x_test)

    g_bar = preds.mean(axis=0)
    std = preds.std(axis=0)
    bias2 = np.mean((g_bar - f(x_test)) ** 2)
    variance = np.mean(np.var(preds, axis=0))

    return {'bias2': bias2, 'variance': variance, 'eout': bias2 + variance,
            'g_bar': g_bar, 'std': std, 'x_test': x_test}

def learning_curve(f, fit_fn, predict_fn, n_list, n_datasets=1500, n_test=500, sigma=0.0):
    """Ein/Eout เฉลี่ยต่อขนาด train set n แต่ละค่าใน n_list"""
    x_test = np.linspace(-1, 1, n_test)
    Ein_list, Eout_list = [], []
    for n in n_list:
        Ein_sum = Eout_sum = 0.0
        for _ in range(n_datasets):
            X = np.random.uniform(-1, 1, n)
            y = f(X) + np.random.normal(0, sigma, n)
            params = fit_fn(X, y)
            Ein_sum += np.mean((predict_fn(params, X) - y) ** 2)
            y_true = f(x_test) + np.random.normal(0, sigma, n_test)
            Eout_sum += np.mean((predict_fn(params, x_test) - y_true) ** 2)
        Ein_list.append(Ein_sum / n_datasets)
        Eout_list.append(Eout_sum / n_datasets)
    return Ein_list, Eout_list


# --------------------------- Run ---------------------------
print("=" * 70)
print("Bias-Variance Decomposition (n=2, x ~ Uniform[-1,1])")
print("=" * 70)

results = {}
for target_name, f in TARGETS.items():
    results[target_name] = {}
    print(f"\n### Target: {target_name} ###")
    for model_name, (fit_fn, predict_fn) in MODELS.items():
        res = simulate_bias_variance(f, fit_fn, predict_fn)
        results[target_name][model_name] = res
        print(f"  {model_name:<20} bias^2={res['bias2']:.4f}  "
              f"variance={res['variance']:.4f}  Eout={res['eout']:.4f}")

# --------------------------- Plot: average fit + variance band ---------------------------
x_plot = np.linspace(-1, 1, 500)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

for row, (target_name, f) in enumerate(TARGETS.items()):
    for col, model_name in enumerate(MODELS):
        ax = axes[row, col]
        res = results[target_name][model_name]

        ax.plot(x_plot, f(x_plot), 'g-', linewidth=2, label='f(x)')
        ax.plot(res['x_test'], res['g_bar'], 'r--', linewidth=2, label='g_bar(x)')
        ax.fill_between(res['x_test'], res['g_bar'] - res['std'], res['g_bar'] + res['std'],
                         color='red', alpha=0.3, label='±1 std')

        for _ in range(15):  # เส้นตัวอย่าง hypothesis จาก dataset ขนาด n=2 (โชว์ variance)
            X = np.random.uniform(-1, 1, 2)
            params = MODELS[model_name][0](X, f(X))
            ax.plot(x_plot, MODELS[model_name][1](params, x_plot), 'k-', alpha=0.15, linewidth=0.5)

        ax.set_title(f"{target_name} | {model_name}", fontsize=9)
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'average_fit.png'), dpi=150)
print('\nSaved: plots/average_fit.png')

# --------------------------- Plot: learning curves (Ein/Eout vs n) ---------------------------
n_list = [2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
noise_levels = [0.0, 0.1, 0.3]
noise_colors = {0.0: '#3B82F6', 0.1: '#10B981', 0.3: '#F59E0B'}

YMAX = {'x^2': 0.6, 'sin(pi*x)': 1.0}
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row')
for row, (target_name, f) in enumerate(TARGETS.items()):
    ymax = YMAX[target_name]
    for col, model_name in enumerate(MODELS):
        ax = axes2[row, col]
        fit_fn, predict_fn = MODELS[model_name]
        for sigma in noise_levels:
            Ein, Eout = learning_curve(f, fit_fn, predict_fn, n_list, sigma=sigma)
            ax.plot(n_list, np.clip(Ein, 0, ymax), '--', color=noise_colors[sigma], alpha=0.7, label=f'Ein σ={sigma}')
            ax.plot(n_list, np.clip(Eout, 0, ymax), '-', color=noise_colors[sigma], alpha=0.7, label=f'Eout σ={sigma}')
        ax.set_xscale('log')
        ax.set_ylim(0, ymax)
        ax.tick_params(labelleft=True)
        ax.set_title(f"{target_name} | {model_name}", fontsize=8)
        ax.set_xlabel('n')
        ax.set_ylabel('Expected Error')
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'learning_curve.png'), dpi=150)
print('Saved: plots/learning_curve.png')
print('Done!')