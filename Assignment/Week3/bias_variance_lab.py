"""
Bias–Variance Decomposition & Learning Curves Lab
===================================================
ต้นฉบับตาม playground ของอาจารย์:
  - Generalization:    https://waranyuwongseree.github.io/generalization-playground/
  - Learning Curve:    https://waranyuwongseree.github.io/learning-curve-playground/

เปรียบเทียบ polynomial models 4 แบบบนปัญหา regression:
  - constant    : h(x) = w0                (degree 0)
  - linear      : h(x) = w0 + w1 x         (degree 1)
  - quadratic   : h(x) = w0 + w1 x + w2 x² (degree 2)
  - cubic       : h(x) = w0 + w1 x + ...   (degree 3)

ฟังก์ชันเป้าหมาย:
  - f(x) = sin(πx)
  - f(x) = x²
  - f(x) = x³

ข้อมูล: x ~ Uniform[-1, 1]
วิธี fit: Normal Equation (numpy.linalg.lstsq)
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------- Paths ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
RESULTS_JSON = os.path.join(BASE_DIR, 'results.json')
os.makedirs(PLOTS_DIR, exist_ok=True)


def safe_filename(name):
    """แปลงชื่อ target ให้เป็นชื่อไฟล์ที่ปลอดภัยทั้งบน Windows/Linux/Mac."""
    for ch, repl in [(' ', '_'), ('^', ''), ('*', ''), ('(', ''), (')', ''),
                      ('<', ''), ('>', ''), (':', ''), ('"', ''),
                      ('/', ''), ('\\', ''), ('|', ''), ('?', '')]:
        name = name.replace(ch, repl)
    return name


# --------------------------- Target Functions ---------------------------
def f_sin(x):
    return np.sin(np.pi * x)


def f_x2(x):
    return x ** 2


def f_x3(x):
    return x ** 3


TARGETS = {
    'sin': {'fn': f_sin, 'label': r'$f(x)=\sin(\pi x)$', 'title': 'sinpix'},
    'x2': {'fn': f_x2, 'label': r'$f(x)=x^2$', 'title': 'x2'},
    'x3': {'fn': f_x3, 'label': r'$f(x)=x^3$', 'title': 'x3'},
}


# --------------------------- Models (Polynomial Regression) ---------------------------
class PolynomialModel:
    """Polynomial regression ผ่าน Normal Equation."""
    def __init__(self, degree, name):
        self.degree = degree
        self.name = name

    def design_matrix(self, X):
        X = np.asarray(X)
        cols = [X ** d for d in range(self.degree + 1)]
        return np.column_stack(cols)

    def fit(self, X, y):
        Phi = self.design_matrix(X)
        w = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return w

    def predict(self, w, X):
        Phi = self.design_matrix(X)
        return Phi @ w


MODELS = [
    PolynomialModel(0, 'constant'),
    PolynomialModel(1, 'linear'),
    PolynomialModel(2, 'quadratic'),
    PolynomialModel(3, 'cubic'),
]


# --------------------------- Simulation Core ---------------------------
def sample_dataset(f, n, sigma=0.0):
    """สุ่มชุดข้อมูลจาก f บน [-1,1] พร้อม noise N(0, sigma²)."""
    X = np.random.uniform(-1, 1, n)
    y = f(X) + np.random.normal(0, sigma, n)
    return X, y


def simulate_bias_variance(f, model, n_samples=2, n_datasets=20000, n_test=300, sigma=0.0):
    """
    Monte Carlo estimate ของ bias², variance, Eout
    Eout = bias² + variance + sigma²
    """
    x_test = np.linspace(-1, 1, n_test)
    preds = np.zeros((n_datasets, n_test))

    for i in range(n_datasets):
        X, y = sample_dataset(f, n_samples, sigma)
        w = model.fit(X, y)
        preds[i] = model.predict(w, x_test)

    g_bar = preds.mean(axis=0)
    bias2_grid = (g_bar - f(x_test)) ** 2
    var_grid = np.var(preds, axis=0)

    # ค่าเฉลี่ยเหนือ x ~ Uniform[-1,1] คือ integral / 2
    bias2 = np.trapezoid(bias2_grid, x_test) / 2.0
    variance = np.trapezoid(var_grid, x_test) / 2.0
    eout = bias2 + variance + sigma ** 2

    return {
        'bias2': float(bias2),
        'variance': float(variance),
        'eout': float(eout),
        'g_bar': g_bar.tolist(),
        'x_test': x_test.tolist(),
    }


def learning_curve(f, model, n_list, n_datasets=3000, n_test=1000, sigma=0.0):
    """คืนค่า Ein และ Eout เฉลี่ยสำหรับแต่ละ n"""
    x_test = np.linspace(-1, 1, n_test)
    Ein_list, Eout_list = [], []

    for n in n_list:
        # โมเดล degree d ต้องการอย่างน้อย d+1 จุด
        if n < model.degree + 1:
            Ein_list.append(None)
            Eout_list.append(None)
            continue

        Ein_sum, Eout_sum = 0.0, 0.0
        for _ in range(n_datasets):
            X, y = sample_dataset(f, n, sigma)
            w = model.fit(X, y)

            # Ein
            y_pred_train = model.predict(w, X)
            Ein_sum += np.mean((y_pred_train - y) ** 2)

            # Eout (เทียบกับ true target + noise ใหม่)
            y_pred_test = model.predict(w, x_test)
            y_true_test = f(x_test) + np.random.normal(0, sigma, n_test)
            Eout_sum += np.mean((y_pred_test - y_true_test) ** 2)

        Ein_list.append(Ein_sum / n_datasets)
        Eout_list.append(Eout_sum / n_datasets)

    return Ein_list, Eout_list


# --------------------------- Main Results ---------------------------
print("=" * 80)
print("Bias-Variance Decomposition (N=2, Uniform[-1,1])")
print("=" * 80)

results = {'bias_variance': {}, 'learning_curve': {}}

for target_key, target_info in TARGETS.items():
    f = target_info['fn']
    print(f"\n### Target: {target_info['label']} ###")
    results['bias_variance'][target_key] = {}

    for model in MODELS:
        sim = simulate_bias_variance(f, model, n_samples=2, sigma=0.0)
        results['bias_variance'][target_key][model.name] = sim
        print(f"  {model.name:12s}: bias²={sim['bias2']:.4f}, variance={sim['variance']:.4f}, Eout={sim['eout']:.4f}")

# --------------------------- Plot 1: Average Fit ---------------------------
print("\n" + "=" * 80)
print("Generating plots...")
print("=" * 80)

x_plot = np.linspace(-1, 1, 500)
n_sample_fits = 40

fig, axes = plt.subplots(len(TARGETS), len(MODELS), figsize=(16, 10), sharex=True, sharey=True)

for row_idx, (target_key, target_info) in enumerate(TARGETS.items()):
    f = target_info['fn']
    for col_idx, model in enumerate(MODELS):
        ax = axes[row_idx, col_idx]
        # วาด target
        ax.plot(x_plot, f(x_plot), color='#6BA38C', linewidth=2.5, label='target')

        # สุ่ม fit หลายชุด
        for _ in range(n_sample_fits):
            X, y = sample_dataset(f, 2, sigma=0.0)
            if len(X) < model.degree + 1:
                continue
            w = model.fit(X, y)
            ax.plot(x_plot, model.predict(w, x_plot), color='gray', alpha=0.25, linewidth=0.8)

        # วาด average fit
        sim = results['bias_variance'][target_key][model.name]
        g_bar = np.array(sim['g_bar'])
        x_test = np.array(sim['x_test'])
        ax.plot(x_test, g_bar, color='#E07A5F', linestyle='--', linewidth=2.5, label=r'$\bar{g}(x)$')

        if row_idx == 0:
            ax.set_title(model.name, fontsize=12, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(target_info['label'], fontsize=12)
        if row_idx == len(TARGETS) - 1:
            ax.set_xlabel('x')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.98))
fig.suptitle('Average Fit: target (green), sample fits (gray), average fit (red dashed)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'average_fit.png'), dpi=150, bbox_inches='tight')
print("Saved: plots/average_fit.png")


# --------------------------- Plot 2: Bias² and Variance Curves ---------------------------
fig, axes = plt.subplots(len(TARGETS), len(MODELS), figsize=(16, 10), sharex=True, sharey=True)

for row_idx, (target_key, target_info) in enumerate(TARGETS.items()):
    f = target_info['fn']
    for col_idx, model in enumerate(MODELS):
        ax = axes[row_idx, col_idx]
        sim = results['bias_variance'][target_key][model.name]
        x_test = np.array(sim['x_test'])

        # recompute sample curves for variance band
        preds = np.zeros((500, len(x_test)))
        for i in range(500):
            X, y = sample_dataset(f, 2, sigma=0.0)
            if len(X) < model.degree + 1:
                continue
            w = model.fit(X, y)
            preds[i] = model.predict(w, x_test)
        g_bar = preds.mean(axis=0)
        std_curve = preds.std(axis=0)

        ax.plot(x_test, f(x_test), color='#6BA38C', linewidth=2, label='target')
        ax.plot(x_test, g_bar, color='#5B8FB9', linewidth=2.5, label=r'$\bar{g}(x)$')
        ax.fill_between(x_test, g_bar - std_curve, g_bar + std_curve,
                        color='#5B8FB9', alpha=0.25, label='±1 std')

        if row_idx == 0:
            ax.set_title(model.name, fontsize=12, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(target_info['label'], fontsize=12)
        if row_idx == len(TARGETS) - 1:
            ax.set_xlabel('x')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.98))
fig.suptitle('Bias-Variance: average fit (blue) ±1 std band vs target (green)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'bias_variance_band.png'), dpi=150, bbox_inches='tight')
print("Saved: plots/bias_variance_band.png")


# --------------------------- Plot 3: Bias²(x) and Var(x) Curves ---------------------------
fig, axes = plt.subplots(len(TARGETS), len(MODELS), figsize=(16, 10), sharex=True, sharey=True)

for row_idx, (target_key, target_info) in enumerate(TARGETS.items()):
    f = target_info['fn']
    for col_idx, model in enumerate(MODELS):
        ax = axes[row_idx, col_idx]
        sim = results['bias_variance'][target_key][model.name]
        x_test = np.array(sim['x_test'])

        # recompute curves
        preds = np.zeros((2000, len(x_test)))
        for i in range(2000):
            X, y = sample_dataset(f, 2, sigma=0.0)
            if len(X) < model.degree + 1:
                continue
            w = model.fit(X, y)
            preds[i] = model.predict(w, x_test)
        g_bar = preds.mean(axis=0)
        bias2_x = (g_bar - f(x_test)) ** 2
        var_x = np.var(preds, axis=0)

        ax.plot(x_test, bias2_x, color='#E07A5F', linewidth=2, label=r'$\mathrm{bias}^2(x)$')
        ax.plot(x_test, var_x, color='#5B8FB9', linewidth=2, label=r'$\mathrm{var}(x)$')

        if row_idx == 0:
            ax.set_title(model.name, fontsize=12, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(target_info['label'], fontsize=12)
        if row_idx == len(TARGETS) - 1:
            ax.set_xlabel('x')
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.98))
fig.suptitle(r'$\mathrm{bias}^2(x)$ (coral) and $\mathrm{var}(x)$ (blue)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'bias2_var_curves.png'), dpi=150, bbox_inches='tight')
print("Saved: plots/bias2_var_curves.png")


# --------------------------- Plot 4-6: Learning Curves ---------------------------
n_list = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
noise_levels = [0.0, 0.1, 0.3]
noise_colors = {0.0: '#3B82F6', 0.1: '#10B981', 0.3: '#F59E0B'}

for target_key, target_info in TARGETS.items():
    f = target_info['fn']
    results['learning_curve'][target_key] = {}

    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4), sharey=False)
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        results['learning_curve'][target_key][model.name] = {}

        for sigma in noise_levels:
            Ein, Eout = learning_curve(f, model, n_list, sigma=sigma, n_datasets=3000)
            results['learning_curve'][target_key][model.name][f'sigma_{sigma}'] = {
                'n': n_list,
                'Ein': [None if e is None else float(e) for e in Ein],
                'Eout': [None if e is None else float(e) for e in Eout],
            }

            valid_n = [n for n, e in zip(n_list, Ein) if e is not None]
            valid_Ein = [e for e in Ein if e is not None]
            valid_Eout = [e for e in Eout if e is not None]

            ax.plot(valid_n, valid_Ein, '--o', color=noise_colors[sigma], label=f'E_in σ={sigma}', alpha=0.8)
            ax.plot(valid_n, valid_Eout, '-s', color=noise_colors[sigma], label=f'E_out σ={sigma}', alpha=0.8)

        ax.set_xlabel('n (number of samples)')
        ax.set_ylabel('Expected Error')
        ax.set_title(model.name)
        ax.set_xscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Learning Curves | Target: {target_info['label']}", fontsize=14)
    plt.tight_layout()
    filename = f"learning_curve_{safe_filename(target_info['title'])}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    print(f"Saved: plots/{filename}")


# --------------------------- Save numerical results ---------------------------
with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved numerical results: {RESULTS_JSON}")


# --------------------------- Summary Table ---------------------------
print("\n" + "=" * 80)
print("Summary: Bias-Variance Decomposition (N=2, σ=0)")
print("=" * 80)
print(f"{'Target':<12} {'Model':<12} {'bias²':<10} {'variance':<10} {'Eout':<10}")
print("-" * 80)
for target_key, target_info in TARGETS.items():
    for model in MODELS:
        r = results['bias_variance'][target_key][model.name]
        print(f"{target_info['title']:<12} {model.name:<12} {r['bias2']:<10.4f} {r['variance']:<10.4f} {r['eout']:<10.4f}")

print("\nDone!")
