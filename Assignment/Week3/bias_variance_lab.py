"""
Bias-Variance Decomposition Lab
================================
เปรียบเทียบ 3 โมเดลบนปัญหา f(x) = sin(pi*x) และ f(x) = x^2
- Constant model        : h(x) = b
- Linear model          : h(x) = w0 + w1*x
- Linear through origin : h(x) = w*x

ข้อมูล: สุ่ม x ~ Uniform[-1, 1] จำนวน n=2 ตัวอย่าง (ยกเว้น learning curve)
วิธี: ใช้ Normal Equation (numpy.linalg.lstsq) ไม่ใช้ Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

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
        # ค่าคงที่ = ค่าเฉลี่ยของ y
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
        # design matrix [1, x]
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
        # design matrix [x]
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        return {'w': w[0]}
    @staticmethod
    def predict(params, X):
        X = np.asarray(X)
        return params['w'] * X

MODELS = [ConstantModel, LinearModel, LinearOriginModel]

# --------------------------- Analytical Helpers ---------------------------
def safe_fit_linear(x1, x2, y1, y2):
    """fit เส้นตรงผ่าน 2 จุด รองรับ input ทั้ง scalar และ array"""
    x1, x2, y1, y2 = np.asarray(x1), np.asarray(x2), np.asarray(y1), np.asarray(y2)
    too_close = np.abs(x2 - x1) < 1e-10
    w1 = np.where(too_close, 0.0, (y2 - y1) / np.where(too_close, 1.0, x2 - x1))
    w0 = np.where(too_close, (y1 + y2) / 2.0, y1 - w1 * x1)
    return w0, w1

def gD_constant(x1, x2, f):
    return (f(x1) + f(x2)) / 2

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
def analytical_constant(f, n_x=50000):
    """
    หาค่า bias^2 และ variance ของ constant model แบบ analytical
    ใช้ numerical integration (trapezoidal rule) บนกริดหนาแน่น
    """
    xs = np.linspace(-1, 1, n_x)
    dx_factor = 1.0 / 2.0  # หาร 2 เพราะความกว้างช่วงคือ 2

    E_f = np.trapezoid(f(xs), xs) * dx_factor
    E_f2 = np.trapezoid(f(xs)**2, xs) * dx_factor

    g_bar = E_f  # ค่าคงที่
    bias2 = np.trapezoid((g_bar - f(xs))**2, xs) * dx_factor
    # var(x) = E_D[g_D^2] - g_bar^2 = 0.5*(E[f^2] + E[f]^2) - E[f]^2 = 0.5*(E[f^2]-E[f]^2)
    var = 0.5 * (E_f2 - E_f**2)
    variance = var  # ไม่ขึ้นกับ x
    return {'bias2': bias2, 'variance': variance, 'eout': bias2 + variance,
            'g_bar': lambda x: np.full(np.asarray(x).shape, g_bar)}

def analytical_model(f, model, n_x=400, n_mc=50000, seed_offset=0):
    """
    หาค่า bias^2 และ variance แบบ numerical integration เหนือ x1, x2
    ใช้ Monte Carlo integration สำหรับ E_D และ Riemann sum สำหรับ E_x
    """
    if model == ConstantModel:
        return analytical_constant(f, n_x=50000)

    rng = np.random.default_rng(123 + seed_offset)
    xs = np.linspace(-1, 1, n_x)
    dx = xs[1] - xs[0]

    # เลือกฟังก์ชัน g_D ตามโมเดล
    if model == LinearModel:
        gD_func = gD_linear
    else:  # LinearOriginModel
        gD_func = gD_linear_origin

    # สุ่มชุดข้อมูล (x1, x2) จำนวน n_mc ชุด
    x1 = rng.uniform(-1, 1, n_mc)
    x2 = rng.uniform(-1, 1, n_mc)

    # คำนวณ g_D(x) สำหรับทุก x และทุกชุดข้อมูล ได้ matrix ขนาด (n_mc, n_x)
    # ทำ vectorize โดย broadcast
    X = xs.reshape(1, -1)          # (1, n_x)
    X1 = x1.reshape(-1, 1)         # (n_mc, 1)
    X2 = x2.reshape(-1, 1)         # (n_mc, 1)
    g_vals = gD_func(X, X1, X2, f)  # (n_mc, n_x)

    g_bar = g_vals.mean(axis=0)        # (n_x,)
    E_g2 = np.mean(g_vals**2, axis=0)  # (n_x,)

    bias2_grid = (g_bar - f(xs))**2
    var_grid = E_g2 - g_bar**2

    # ประมาณอินทิเกรตบน [-1,1] ด้วย trapezoidal rule
    bias2 = np.trapezoid(bias2_grid, xs) / 2.0
    variance = np.trapezoid(var_grid, xs) / 2.0

    # สร้าง interpolator ง่าย ๆ สำหรับ g_bar(x)
    def g_bar_func(x_query):
        return np.interp(np.asarray(x_query), xs, g_bar)

    return {'bias2': bias2, 'variance': variance, 'eout': bias2 + variance, 'g_bar': g_bar_func}

# --------------------------- Simulation Bias/Variance ---------------------------
def simulate_bias_variance(f, model, n_samples=2, n_datasets=20000, n_test=2000):
    """Monte Carlo estimate ของ bias^2 และ variance"""
    x_test = np.linspace(-1, 1, n_test)
    preds = np.zeros((n_datasets, n_test))

    for i in range(n_datasets):
        X = np.random.uniform(-1, 1, n_samples)
        # ถ้าใส่ noise จะถูกจัดการที่ฟังก์ชันอื่น ที่นี่ไม่ใส่ noise
        y = f(X)
        params = model.fit(X, y)
        preds[i] = model.predict(params, x_test)

    g_bar = preds.mean(axis=0)
    bias2 = np.mean((g_bar - f(x_test))**2)
    variance = np.mean(np.var(preds, axis=0))
    eout = bias2 + variance
    return {'bias2': bias2, 'variance': variance, 'eout': eout, 'g_bar': g_bar, 'x_test': x_test}

# --------------------------- Learning Curves ---------------------------
def learning_curve(f, model, n_list, n_datasets=3000, n_test=1000, sigma=0.0):
    """คืนค่า Ein และ Eout เฉลี่ยสำหรับแต่ละ n"""
    x_test = np.linspace(-1, 1, n_test)
    Ein_list, Eout_list = [], []

    for n in n_list:
        Ein_sum, Eout_sum = 0.0, 0.0
        for _ in range(n_datasets):
            X = np.random.uniform(-1, 1, n)
            y = f(X) + np.random.normal(0, sigma, n)
            params = model.fit(X, y)

            # Ein
            y_pred_train = model.predict(params, X)
            Ein_sum += np.mean((y_pred_train - y)**2)

            # Eout
            y_pred_test = model.predict(params, x_test)
            y_true_test = f(x_test) + np.random.normal(0, sigma, n_test)
            Eout_sum += np.mean((y_pred_test - y_true_test)**2)

        Ein_list.append(Ein_sum / n_datasets)
        Eout_list.append(Eout_sum / n_datasets)

    return np.array(Ein_list), np.array(Eout_list)

# --------------------------- Main Results ---------------------------
print("=" * 80)
print("Bias-Variance Decomposition (N=2, Uniform[-1,1])")
print("=" * 80)

results = {}
for target_name, f in TARGETS.items():
    print(f"\n### Target: {target_name} ###")
    results[target_name] = {}
    for model in MODELS:
        # Analytical
        ana = analytical_model(f, model)
        # Simulation
        sim = simulate_bias_variance(f, model)
        results[target_name][model.name] = {'analytical': ana, 'simulation': sim}

        print(f"\nModel: {model.name}")
        print(f"  Analytical -> bias^2 = {ana['bias2']:.4f}, variance = {ana['variance']:.4f}, Eout = {ana['eout']:.4f}")
        print(f"  Simulation -> bias^2 = {sim['bias2']:.4f}, variance = {sim['variance']:.4f}, Eout = {sim['eout']:.4f}")

# --------------------------- Plotting ---------------------------
# สร้างโฟลเดอร์ plots สำหรับเก็บภาพที่ visualize ผลลัพธ์
os.makedirs('plots', exist_ok=True)

def safe_filename(name):
    """แปลงชื่อ target ให้เป็นชื่อไฟล์ที่ปลอดภัยทั้งบน Windows/Linux/Mac
    (Windows ห้ามใช้ < > : " / \\ | ? * ในชื่อไฟล์)"""
    for ch, repl in [(' ', '_'), ('^', ''), ('*', ''), ('(', ''), (')', ''),
                      ('<', ''), ('>', ''), (':', ''), ('"', ''),
                      ('/', ''), ('\\', ''), ('|', ''), ('?', '')]:
        name = name.replace(ch, repl)
    return name


# =============================================================================
# Plot 1: Average Fit Visualization (paired: raw hypothesis lines + variance band)
# =============================================================================
# สำหรับแต่ละ (target, model) วาด 2 กราฟคู่กัน:
#   - กราฟซ้าย : เส้นเป้าหมาย f(x) (เขียว) + เส้น hypothesis จากชุดข้อมูลสุ่มหลายชุด (เทาบาง ๆ)
#                + เส้นเฉลี่ย g_bar(x) (แดงประ) -> ให้เห็น "variance" เป็นความกระจายของเส้นเทา
#   - กราฟขวา  : เส้นเดียวกัน แต่แสดง variance เป็นแถบสีแดงโปร่งแสง (mean ± std) แทนเส้นเทา
# ใส่ noise สุ่ม (Gaussian, std = NOISE_STD) ลงในตัวอย่างตอนสร้าง hypothesis เพื่อจำลอง
# สถานการณ์ noisy target ตามที่โจทย์ข้อ 3 ให้ทดลองเพิ่มเติม
# =============================================================================
NOISE_STD = 0  # ระดับ noise ที่ใส่ตอน fit ตัวอย่างสำหรับภาพนี้ (ปรับได้)

MODEL_TITLE = {
    'Constant': 'constant',
    'Linear': 'linear',
    'Linear through origin': 'linear_origin',
}
TARGET_TITLE = {
    'sin(pi*x)': r'$\sin(\pi x)$',
    'x^2': r'$x^2$',
}

def simulate_band(f, model, sigma, n_samples=2, n_datasets=20000, n_test=300, lo_pct=25, hi_pct=85):
    """หา g_bar(x) (mean) และแถบการกระจาย (lo/hi percentile) ของ hypothesis จาก simulation
    ใช้ percentile แทน mean +/- std เพราะ linear model แบบ fit 2 จุด มีโอกาสได้ slope ที่ชันมาก
    เวลา x1,x2 สุ่มมาใกล้กัน (denominator ~ 0) ทำให้ std ถูกดึงสูงผิดปกติจนแถบเต็มกราฟ (heavy-tailed)
    percentile ทนต่อ outlier แบบนี้ได้ดีกว่า และให้แถบที่สะท้อน "การกระจายทั่วไป" ได้ตรงกว่า"""
    x_test = np.linspace(-1, 1, n_test)
    preds = np.zeros((n_datasets, n_test))
    for i in range(n_datasets):
        X = np.random.uniform(-1, 1, n_samples)
        y = f(X) + np.random.normal(0, sigma, n_samples)
        params = model.fit(X, y)
        preds[i] = model.predict(params, x_test)
    mean_curve = preds.mean(axis=0)
    lo_curve = np.percentile(preds, lo_pct, axis=0)
    hi_curve = np.percentile(preds, hi_pct, axis=0)
    return x_test, mean_curve, lo_curve, hi_curve

x_plot = np.linspace(-1, 1, 500)
fig, axes = plt.subplots(2, 6, figsize=(20, 7), sharex=True, sharey=True)

for row_idx, (target_name, f) in enumerate(TARGETS.items()):
    for m_idx, model in enumerate(MODELS):
        ax_lines = axes[row_idx, m_idx * 2]
        ax_band = axes[row_idx, m_idx * 2 + 1]
        title = f"{TARGET_TITLE[target_name]} — Model: {MODEL_TITLE[model.name]}"

        x_test, mean_curve, lo_curve, hi_curve = simulate_band(f, model, sigma=NOISE_STD)

        for ax in (ax_lines, ax_band):
            ax.plot(x_plot, f(x_plot), color='green', linewidth=2)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(-1, 1)
            ax.grid(True, alpha=0.3)

        # กราฟซ้าย: เส้น hypothesis รายชุดข้อมูล (แสดง variance เป็นความกระจายของเส้น)
        for _ in range(60):
            X = np.random.uniform(-1, 1, 2)
            y = f(X) + np.random.normal(0, NOISE_STD, 2)
            params = model.fit(X, y)
            ax_lines.plot(x_plot, model.predict(params, x_plot), color='black', alpha=0.15, linewidth=0.6)
        ax_lines.plot(x_test, mean_curve, color='red', linestyle='--', linewidth=2)

        # กราฟขวา: variance เป็นแถบสีแดง (10th-90th percentile, กันแถบระเบิดจาก outlier)
        ax_band.plot(x_test, mean_curve, color='red', linestyle='--', linewidth=2)
        ax_band.fill_between(x_test, np.clip(lo_curve, -2, 2), np.clip(hi_curve, -2, 2),
                              color='red', alpha=0.3)

for ax in axes[:, 0]:
    ax.set_ylabel('y')
for ax in axes[-1, :]:
    ax.set_xlabel('x')

axes[0, 0].set_ylim(-2, 2)

target_handle = plt.Line2D([0], [0], color='green', lw=2, label='Target')
average_handle = plt.Line2D([0], [0], color='red', lw=2, ls='--', label='Average')
fig.legend(handles=[target_handle, average_handle], loc='upper center', ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, 1.03))
fig.suptitle(f"Bias-Variance Decomposition (Two Samples, Noise Std: {NOISE_STD})", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('plots/average_fit.png', dpi=150)
print("\nSaved: plots/average_fit.png")

# 2. ตารางเปรียบเทียบ
print("\n" + "=" * 80)
print("Summary Table")
print("=" * 80)
print(f"{'Target':<12} {'Model':<22} {'bias^2 (ana)':<14} {'var (ana)':<12} {'Eout (ana)':<12} {'Eout (sim)':<12}")
print("-" * 80)
for target_name in TARGETS:
    for model in MODELS:
        ana = results[target_name][model.name]['analytical']
        sim = results[target_name][model.name]['simulation']
        print(f"{target_name:<12} {model.name:<22} {ana['bias2']:<14.4f} {ana['variance']:<12.4f} {ana['eout']:<12.4f} {sim['eout']:<12.4f}")

# =============================================================================
# Plot 2 & 3: Learning Curves
# =============================================================================
# ภาพนี้แสดงค่า Ein (training error) และ Eout (generalization error)
# เมื่อจำนวนตัวอย่าง n เพิ่มขึ้น สำหรับแต่ละโมเดลและแต่ละระดับ noise σ
#
# ที่ต้องการสังเกต:
#   - ที่ n=2 Linear model มี Ein ≈ 0 (เส้นตรงผ่าน 2 จุดพอดี)
#     แต่ Eout อาจสูงมาก โดยเฉพาะเมื่อมี noise (overfitting)
#   - เมื่อ n เพิ่ม Ein มักจะเพิ่ม (ยากที่จะ fit ทุกจุด) แต่ Eout ลดลง
#     เพราะ variance ลด
#   - ระดับ noise สูงขึ้นทำให้ Eout สูงขึ้นประมาณ σ²
#   - Constant model มีเส้นโค้งที่นิ่งกว่า สะท้อนว่า variance ต่ำ
#   - แต่ละกราฟมี y-axis เป็นของตัวเอง เพราะ scale ของ error ต่างกันมาก
#     (Linear model กับ noise สูงที่ n=2 มี Eout เป็นพัน)
# =============================================================================
n_list = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
noise_levels = [0.0, 0.1, 0.3]

for target_name, f in TARGETS.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for sigma in noise_levels:
            Ein, Eout = learning_curve(f, model, n_list, sigma=sigma)
            label = f"σ={sigma}"
            ax.plot(n_list, Ein, '--o', label=f"Ein {label}", alpha=0.7)
            ax.plot(n_list, Eout, '-s', label=f"Eout {label}", alpha=0.7)
        ax.set_xlabel('n (number of samples)')
        ax.set_ylabel('Expected Error')
        ax.set_title(f"{model.name}")
        ax.set_xscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Learning Curves | Target: {target_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/learning_curve_{safe_filename(target_name)}.png', dpi=150)
    print(f"Saved: plots/learning_curve_{safe_filename(target_name)}.png")

print("\nDone!")

# --------------------------- Show all figures interactively ---------------------------
# เปิดหน้าต่างแสดงกราฟทั้งหมดทันทีที่รันโปรแกรม (นอกเหนือจากการ save ไฟล์ไว้ใน plots/)
plt.show()