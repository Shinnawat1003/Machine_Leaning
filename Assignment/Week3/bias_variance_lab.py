"""
Bias-Variance Decomposition Lab
================================
โจทย์: หาค่าความเอนเอียง (bias²) และความแปรปรวน (variance)
ด้วย analytical method และ simulation สำหรับ:
  1. Constant model        : h(x) = b
  2. Linear model          : h(x) = w0 + w1*x
  3. Linear through origin : h(x) = w*x

ฟังก์ชันเป้าหมาย:
  1. f(x) = sin(pi*x)
  2. f(x) = x^2

ข้อมูล: x ~ Uniform[-1, 1] จำนวน n=2 ตัวอย่าง (ยกเว้น learning curve)
วิธี fit: Normal Equation (numpy.linalg.lstsq)
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)  # ล็อค seed เพื่อให้ผลการสุ่มทดซ้ำได้ (reproducible)

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


TARGETS = {
    'sin(pi*x)': f_sin,
    'x^2': f_x2,
}


# --------------------------- Models (Normal Equation) ---------------------------
class ConstantModel:
    name = 'Constant'

    @staticmethod
    def fit(X, y):
        # ค่า b ที่ลด MSE ที่สุดคือค่าเฉลี่ยของ y
        return {'b': np.mean(y)}

    @staticmethod
    def predict(params, X):
        X = np.asarray(X)
        return np.full(X.shape, params['b'])


class LinearModel:
    name = 'Linear'

    @staticmethod
    def fit(X, y):
        X = np.asarray(X)
        # design matrix: คอลัมน์แรกเป็น 1 (bias/w0), คอลัมน์ที่สองเป็น x (w1)
        Phi = np.column_stack([np.ones(len(X)), X])
        w = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return {'w0': w[0], 'w1': w[1]}

    @staticmethod
    def predict(params, X):
        X = np.asarray(X)
        return params['w0'] + params['w1'] * X


class LinearOriginModel:
    name = 'Linear through origin'

    @staticmethod
    def fit(X, y):
        X = np.asarray(X).reshape(-1, 1)
        # ไม่มี bias term: จึง fit แค่ w*x ผ่าน origin เท่านั้น
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        return {'w': w[0]}

    @staticmethod
    def predict(params, X):
        X = np.asarray(X)
        return params['w'] * X


MODELS = [ConstantModel, LinearModel, LinearOriginModel]  # รายชื่อ model ทั้งหมด


# --------------------------- Analytical Helpers ---------------------------
def safe_fit_linear(x1, x2, y1, y2):
    """fit เส้นตรงผ่าน 2 จุด รองรับ input ทั้ง scalar และ array"""
    x1, x2, y1, y2 = np.asarray(x1), np.asarray(x2), np.asarray(y1), np.asarray(y2)
    too_close = np.abs(x2 - x1) < 1e-10
    w1 = np.where(too_close, 0.0, (y2 - y1) / np.where(too_close, 1.0, x2 - x1))
    w0 = np.where(too_close, (y1 + y2) / 2.0, y1 - w1 * x1)
    return w0, w1


def gD_linear(x, x1, x2, f):
    y1, y2 = f(x1), f(x2)
    w0, w1 = safe_fit_linear(x1, x2, y1, y2)
    return w0 + w1 * x


def gD_linear_origin(x, x1, x2, f):
    y1, y2 = f(x1), f(x2)
    denom = x1**2 + x2**2
    too_small = denom < 1e-10
    w = np.where(too_small, 0.0, (x1 * y1 + x2 * y2) / np.where(too_small, 1.0, denom))
    return w * x


# --------------------------- Analytical Bias/Variance ---------------------------
def analytical_constant(f, n_x=200000):
    """
    หาค่า bias² และ variance ของ constant model แบบ analytical
    ใช้ numerical integration (trapezoidal rule) บนกริดหนาแน่น
    n_x = 200,000 จุด: กริดละเอียดมากเพื่อประมาณค่าอินทิกรัล
    """
    xs = np.linspace(-1, 1, n_x)
    dx_factor = 1.0 / 2.0  # หาร 2 เพราะความกว้างช่วงคือ 2 (pdf ของ Uniform[-1,1] = 1/2)

    E_f = np.trapezoid(f(xs), xs) * dx_factor      # E_x[f(x)]
    E_f2 = np.trapezoid(f(xs)**2, xs) * dx_factor  # E_x[f(x)^2]

    g_bar = E_f
    bias2 = np.trapezoid((g_bar - f(xs))**2, xs) * dx_factor
    # variance = E_D[g_D^2] - g_bar^2
    # กับ constant model และ n=2: E_D[g_D^2] = 0.5*(E[f^2] + E[f]^2)
    # ดังนั้น variance = 0.5*(E[f^2] - E[f]^2)
    var = 0.5 * (E_f2 - E_f**2)
    variance = var

    return {
        'bias2': float(bias2),
        'variance': float(variance),
        'eout': float(bias2 + variance),
        'g_bar': lambda x: np.full(np.asarray(x).shape, g_bar),
    }


def analytical_model(f, model, n_x=400, n_mc=200000, seed_offset=0):
    """
    หาค่า bias² และ variance แบบ numerical integration เหนือ x1, x2
    ใช้ Monte Carlo integration สำหรับ E_D และ Riemann sum สำหรับ E_x
    n_x   = 400:    กริดสำหรับประมาณค่า E_x
    n_mc  = 200,000: จำนวนชุดข้อมูล (x1,x2) สำหรับ Monte Carlo ประมาณ E_D
    """
    if model == ConstantModel:
        return analytical_constant(f, n_x=200000)

    rng = np.random.default_rng(123 + seed_offset)
    xs = np.linspace(-1, 1, n_x)

    if model == LinearModel:
        gD_func = gD_linear          # fit เส้นตรงผ่าน 2 จุด
    else:  # LinearOriginModel
        gD_func = gD_linear_origin   # fit เส้นตรงผ่าน origin จาก 2 จุด

    x1 = rng.uniform(-1, 1, n_mc)
    x2 = rng.uniform(-1, 1, n_mc)

    X = xs.reshape(1, -1)
    X1 = x1.reshape(-1, 1)
    X2 = x2.reshape(-1, 1)
    g_vals = gD_func(X, X1, X2, f)

    g_bar = g_vals.mean(axis=0)
    E_g2 = np.mean(g_vals**2, axis=0)

    bias2_grid = (g_bar - f(xs))**2
    var_grid = E_g2 - g_bar**2

    bias2 = np.trapezoid(bias2_grid, xs) / 2.0
    variance = np.trapezoid(var_grid, xs) / 2.0

    def g_bar_func(x_query):
        return np.interp(np.asarray(x_query), xs, g_bar)

    return {
        'bias2': float(bias2),
        'variance': float(variance),
        'eout': float(bias2 + variance),
        'g_bar': g_bar_func,
    }


# --------------------------- Simulation Bias/Variance ---------------------------
def simulate_bias_variance(f, model, n_samples=2, n_datasets=50000, n_test=300, sigma=0.0):
    """
    Monte Carlo estimate ของ bias² และ variance
    n_samples  = 2:      ขนาดชุด train (ยกเว้น learning curve)
    n_datasets = 50,000: จำนวนชุดข้อมูล D สำหรับประมาณ E_D
    n_test     = 300:    จำนวนจุดทดสอบบนช่วง [-1,1]
    sigma      = 0.0:    ระดับ noise (0 = ไม่มี noise)
    """
    x_test = np.linspace(-1, 1, n_test)
    preds = np.zeros((n_datasets, n_test))

    for i in range(n_datasets):
        X = np.random.uniform(-1, 1, n_samples)
        y = f(X) + np.random.normal(0, sigma, n_samples)  # เพิ่ม noise ตาม sigma
        params = model.fit(X, y)
        preds[i] = model.predict(params, x_test)

    g_bar = preds.mean(axis=0)
    std = preds.std(axis=0)
    bias2 = np.mean((g_bar - f(x_test))**2)
    variance = np.mean(np.var(preds, axis=0))
    eout = bias2 + variance + sigma**2

    return {
        'bias2': float(bias2),
        'variance': float(variance),
        'eout': float(eout),
        'g_bar': g_bar.tolist(),
        'std': std.tolist(),
        'x_test': x_test.tolist(),
    }


# --------------------------- Learning Curves ---------------------------
def learning_curve(f, model, n_list, n_datasets=3000, n_test=1000, sigma=0.0):
    """
    คืนค่า Ein และ Eout เฉลี่ยสำหรับแต่ละ n
    n_datasets = 3,000: จำนวนชุดข้อมูลเฉลี่ยสำหรับแต่ละค่า n
    n_test     = 1,000: จำนวนจุดทดสอบสำหรับคำนวณ Eout
    """
    x_test = np.linspace(-1, 1, n_test)
    Ein_list, Eout_list = [], []

    for n in n_list:
        Ein_sum, Eout_sum = 0.0, 0.0
        for _ in range(n_datasets):
            X = np.random.uniform(-1, 1, n)
            y = f(X) + np.random.normal(0, sigma, n)
            params = model.fit(X, y)

            y_pred_train = model.predict(params, X)
            Ein_sum += np.mean((y_pred_train - y)**2)

            y_pred_test = model.predict(params, x_test)
            y_true_test = f(x_test) + np.random.normal(0, sigma, n_test)
            Eout_sum += np.mean((y_pred_test - y_true_test)**2)

        Ein_list.append(Ein_sum / n_datasets)
        Eout_list.append(Eout_sum / n_datasets)

    return Ein_list, Eout_list


# --------------------------- Main Results ---------------------------
print("=" * 80)
print("Bias-Variance Decomposition (N=2, Uniform[-1,1])")
print("=" * 80)

results = {'bias_variance': {}}

for target_name, f in TARGETS.items():
    print(f"\n### Target: {target_name} ###")
    results['bias_variance'][target_name] = {}
    for model in MODELS:
        # คำนวณทั้งแบบ analytical และ simulation สำหรับทุก model
        ana = analytical_model(f, model)
        sim = simulate_bias_variance(f, model)
        results['bias_variance'][target_name][model.name] = {
            'analytical': {
                'bias2': ana['bias2'],
                'variance': ana['variance'],
                'eout': ana['eout'],
            },
            'simulation': sim,
        }

        print(f"\nModel: {model.name}")
        print(f"  Analytical -> bias^2 = {ana['bias2']:.4f}, variance = {ana['variance']:.4f}, Eout = {ana['eout']:.4f}")
        print(f"  Simulation -> bias^2 = {sim['bias2']:.4f}, variance = {sim['variance']:.4f}, Eout = {sim['eout']:.4f}")


# --------------------------- Plotting ---------------------------
print("\n" + "=" * 80)
print("Generating plots...")
print("=" * 80)

x_plot = np.linspace(-1, 1, 500)  # กริด 500 จุดสำหรับวาดกราฟ

# =============================================================================
# Plot 1: Average Fit Visualization
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

for row_idx, (target_name, f) in enumerate(TARGETS.items()):
    for col_idx, model in enumerate(MODELS):
        ax = axes[row_idx, col_idx]
        ax.plot(x_plot, f(x_plot), 'g-', linewidth=2, label='Target f(x)')

        # วาด g_bar (ค่าเฉลี่ยของ hypothesis ทั้งหมดบน test set) จาก simulation
        res = results['bias_variance'][target_name][model.name]['simulation']
        g_bar_arr = np.array(res['g_bar'])
        std_arr = np.array(res['std'])
        ax.plot(res['x_test'], g_bar_arr, 'r--', linewidth=2, label='g_bar(x) (sim)')

        # แถบสีแดงแสดงช่วง ±1 std รอบๆ g_bar(x)
        ax.fill_between(res['x_test'], g_bar_arr - std_arr, g_bar_arr + std_arr,
                        color='red', alpha=0.3, label='±1 std')

        # ไฮไลค่าเฉลี่ยของ f(x) บนช่วง [-1, 1]
        f_avg = np.trapezoid(f(x_plot), x_plot) / (x_plot[-1] - x_plot[0])
        ax.axhline(y=f_avg, color='darkred', alpha=0.5, linewidth=1.5, linestyle='-', label='E[f(x)]')

        # วาด hypothesis ตัวอย่าง 20 เส้น จากชุด train ขนาด n=2 เพื่อแสดง variance
        for _ in range(20):
            X = np.random.uniform(-1, 1, 2)
            y = f(X)
            params = model.fit(X, y)
            ax.plot(x_plot, model.predict(params, x_plot), 'k-', alpha=0.2, linewidth=0.5)

        ax.set_title(f"{target_name} | {model.name}")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'average_fit.png'), dpi=150)
print("\nSaved: plots/average_fit.png")


# ตารางเปรียบเทียบ
print("\n" + "=" * 80)
print("Summary Table")
print("=" * 80)
print(f"{'Target':<12} {'Model':<22} {'bias^2 (ana)':<14} {'var (ana)':<12} {'Eout (ana)':<12} {'Eout (sim)':<12}")
print("-" * 80)
for target_name in TARGETS:
    for model in MODELS:
        ana = results['bias_variance'][target_name][model.name]['analytical']
        sim = results['bias_variance'][target_name][model.name]['simulation']
        print(f"{target_name:<12} {model.name:<22} {ana['bias2']:<14.4f} {ana['variance']:<12.4f} {ana['eout']:<12.4f} {sim['eout']:<12.4f}")


# =============================================================================
# Plot 2 & 3: Learning Curves
# =============================================================================
n_list = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]  # ขนาดชุด train ที่ทดลอง
noise_levels = [0.0, 0.1, 0.3]                       # ระดับ noise σ ที่ทดลอง
noise_colors = {0.0: '#3B82F6', 0.1: '#10B981', 0.3: '#F59E0B'}

learning_results = {}

for target_name, f in TARGETS.items():
    learning_results[target_name] = {}
    # sharey=True เพื่อให้ทั้ง 3 model ใช้สเกลแกน y เดียวกัน
    # สำหรับ x^2 ใช้สูงสุด 0.7 ส่วน sin(pi*x) ใช้ 1.0
    y_max = 0.7 if target_name == 'x^2' else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        learning_results[target_name][model.name] = {}
        for sigma in noise_levels:
            Ein, Eout = learning_curve(f, model, n_list, sigma=sigma)
            learning_results[target_name][model.name][f'sigma_{sigma}'] = {
                'n': n_list,
                'Ein': [float(e) for e in Ein],
                'Eout': [float(e) for e in Eout],
            }
            # ตัดค่าที่เกิน y_max ออกจากการพล็อต เพื่อไม่ให้กราฟบานออกไปถึง 1000
            Ein_plot = np.clip(Ein, 0, y_max)
            Eout_plot = np.clip(Eout, 0, y_max)
            label = f"σ={sigma}"
            ax.plot(n_list, Ein_plot, '--', color=noise_colors[sigma], label=f"Ein {label}", alpha=0.7)
            ax.plot(n_list, Eout_plot, '-', color=noise_colors[sigma], label=f"Eout {label}", alpha=0.7)
        ax.set_xlabel('n (number of samples)')
        ax.set_ylabel('Expected Error')
        ax.set_title(f"{model.name}")
        ax.set_xscale('log')
        ax.set_ylim(0, y_max)  # ล็อคแกน y ทั้ง 3 model ให้อยู่ที่ 0 ถึง y_max
        ax.tick_params(labelleft=True)  # แสดงตัวเลขแกน y ในทุก subplot
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Learning Curves | Target: {target_name}", fontsize=14)
    plt.tight_layout()
    filename = f'learning_curve_{safe_filename(target_name)}.png'
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    print(f"Saved: plots/{filename}")


# --------------------------- Save numerical results ---------------------------
results['learning_curve'] = learning_results
with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved numerical results: {RESULTS_JSON}")

print("\nDone!")
