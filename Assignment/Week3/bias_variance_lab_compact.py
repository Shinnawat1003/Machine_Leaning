import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def safe_filename(name):
    for ch in ' ^*()<>?:"/\\|?':
        name = name.replace(ch, '')
    return name.replace(' ', '_')


TARGETS = {'sin(pi*x)': lambda x: np.sin(np.pi * x), 'x^2': lambda x: x ** 2}
MODELS = ['Constant', 'Linear', 'Linear through origin']
NOISE = [0.0, 0.1, 0.3]
COLORS = {0.0: '#3B82F6', 0.1: '#10B981', 0.3: '#F59E0B'}
N_LIST = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
YMAX = {'x^2': 0.6, 'sin(pi*x)': 1.0}


def fit_predict(model, X, y, xq):
    X, y, xq = map(np.asarray, (X, y, xq))
    if model == 'Constant':
        return np.full_like(xq, np.mean(y))
    if model == 'Linear':
        w = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        return w[0] + w[1] * xq
    return np.linalg.lstsq(X.reshape(-1, 1), y, rcond=None)[0] * xq


def simulate(f, model, n_datasets=50000, n_test=300):
    x_test = np.linspace(-1, 1, n_test)
    preds = np.zeros((n_datasets, n_test))
    for i in range(n_datasets):
        X = np.random.uniform(-1, 1, 2)
        preds[i] = fit_predict(model, X, f(X), x_test)
    g_bar, std = preds.mean(axis=0), preds.std(axis=0)
    bias2 = np.mean((g_bar - f(x_test)) ** 2)
    variance = np.mean(np.var(preds, axis=0))
    return {'bias2': float(bias2), 'variance': float(variance), 'eout': float(bias2 + variance),
            'g_bar': g_bar.tolist(), 'std': std.tolist(), 'x_test': x_test.tolist()}


def learning_curve(f, model, n_list, sigma=0.0, n_datasets=3000, n_test=1000):
    x_test = np.linspace(-1, 1, n_test)
    Ein, Eout = [], []
    for n in n_list:
        ein_sum = eout_sum = 0.0
        for _ in range(n_datasets):
            X = np.random.uniform(-1, 1, n)
            y = f(X) + np.random.normal(0, sigma, n)
            ein_sum += np.mean((fit_predict(model, X, y, X) - y) ** 2)
            eout_sum += np.mean((fit_predict(model, X, y, x_test) - f(x_test) + np.random.normal(0, sigma, n_test)) ** 2)
        Ein.append(ein_sum / n_datasets)
        Eout.append(eout_sum / n_datasets)
    return Ein, Eout


print('=' * 80)
print('Summary Table')
print('=' * 80)
print(f"{'Target':<12} {'Model':<22} {'bias^2':<10} {'variance':<10} {'Eout':<10}")
print('-' * 80)

x_plot = np.linspace(-1, 1, 500)
fig_avg, axes_avg = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

for row, (target_name, f) in enumerate(TARGETS.items()):
    for col, model in enumerate(MODELS):
        sim = simulate(f, model)
        print(f"{target_name:<12} {model:<22} {sim['bias2']:<10.4f} {sim['variance']:<10.4f} {sim['eout']:<10.4f}")

        ax = axes_avg[row, col]
        ax.plot(x_plot, f(x_plot), 'g-', linewidth=2, label='Target f(x)')
        g_bar, std = np.array(sim['g_bar']), np.array(sim['std'])
        ax.plot(sim['x_test'], g_bar, 'r--', linewidth=2, label='g_bar(x)')
        ax.fill_between(sim['x_test'], g_bar - std, g_bar + std,
                        color='red', alpha=0.3, label='±1 std')
        for _ in range(20):
            X = np.random.uniform(-1, 1, 2)
            ax.plot(x_plot, fit_predict(model, X, f(X), x_plot), 'k-', alpha=0.2, linewidth=0.5)
        ax.set_title(f"{target_name} | {model}")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'average_fit.png'), dpi=150)
print('\nSaved: plots/average_fit.png')

for target_name, f in TARGETS.items():
    ymax = YMAX[target_name]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        for sigma in NOISE:
            Ein, Eout = learning_curve(f, model, N_LIST, sigma=sigma)
            ax.plot(N_LIST, np.clip(Ein, 0, ymax), '--', color=COLORS[sigma], label=f'Ein σ={sigma}', alpha=0.7)
            ax.plot(N_LIST, np.clip(Eout, 0, ymax), '-', color=COLORS[sigma], label=f'Eout σ={sigma}', alpha=0.7)
        ax.set_xlabel('n (number of samples)')
        ax.set_ylabel('Expected Error')
        ax.set_title(model)
        ax.set_xscale('log')
        ax.set_ylim(0, ymax)
        ax.tick_params(labelleft=True)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Learning Curves | Target: {target_name}', fontsize=14)
    plt.tight_layout()
    filename = f'learning_curve_{safe_filename(target_name)}.png'
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    print(f"Saved: plots/{filename}")

print('\nDone!')
