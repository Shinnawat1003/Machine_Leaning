# Assignment 2: Performance Estimation — resubstitution / holdout / k-fold CV
# target f(x)=sin(pi*x), x~U(-1,1), y=f(x)+noise ; models: Constant, Linear
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUT, exist_ok=True)

f = lambda x: np.sin(np.pi * x)
MODELS = ['Constant', 'Linear']
XG = np.linspace(-1, 1, 4000)          # grid for a precise "true" E_out


def gen_data(n, sigma):
    X = np.random.uniform(-1, 1, n)
    return X, f(X) + np.random.normal(0, sigma, n)


def fit_predict(model, X, y, xq):
    if model == 'Constant':
        return np.full_like(xq, y.mean())
    w = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
    return w[0] + w[1] * xq


def true_eout(model, X, y, sigma):
    return np.mean((fit_predict(model, X, y, XG) - f(XG)) ** 2) + sigma ** 2


def resub(model, X, y):
    return np.mean((fit_predict(model, X, y, X) - y) ** 2)


def holdout(model, X, y, frac=0.7):
    n = len(X); idx = np.random.permutation(n)
    nt = min(max(int(round(frac * n)), 1), n - 1)
    tr, te = idx[:nt], idx[nt:]
    return np.mean((fit_predict(model, X[tr], y[tr], X[te]) - y[te]) ** 2)


def kfold(model, X, y, k=5):
    n = len(X); k = max(2, min(k, n))
    folds = np.array_split(np.random.permutation(n), k)
    errs = [np.mean((fit_predict(model, X[tr := np.concatenate([folds[j] for j in range(k) if j != i])],
                                  y[tr], X[folds[i]]) - y[folds[i]]) ** 2) for i in range(k)]
    return np.mean(errs)


def run(model, n, sigma, frac=0.7, k=5, reps=2000):
    rows = []
    for _ in range(reps):
        X, y = gen_data(n, sigma)
        rows.append((true_eout(model, X, y, sigma), resub(model, X, y),
                      holdout(model, X, y, frac), kfold(model, X, y, k)))
    return pd.DataFrame(rows, columns=['true', 'Resub', 'Holdout', 'KFold'])


def bias_var(df):
    diff = df[['Resub', 'Holdout', 'KFold']].sub(df['true'], axis=0)
    return pd.DataFrame({'bias': diff.mean(), 'var': df[['Resub', 'Holdout', 'KFold']].var(),
                          'mse': (diff ** 2).mean()})


# ---------- 1) single dataset ----------
print('### 1) Single dataset: estimate vs true E_out ###')
for m in MODELS:
    X, y = gen_data(20, 0.3)
    print(f"{m:9s} true={true_eout(m,X,y,0.3):.3f}  resub={resub(m,X,y):.3f}  "
          f"holdout={holdout(m,X,y):.3f}  kfold={kfold(m,X,y):.3f}")

# ---------- 2) bias / variance / mse ----------
print('\n### 2) Bias / Variance / MSE over many datasets ###')
dfs = {m: run(m, 20, 0.3, reps=2000) for m in MODELS}
for m in MODELS:
    print(f'\n{m}\n{bias_var(dfs[m]).round(4)}')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, m in zip(axes, MODELS):
    err = dfs[m][['Resub', 'Holdout', 'KFold']].sub(dfs[m]['true'], axis=0)
    ax.boxplot(err.values, tick_labels=err.columns, showmeans=True)
    ax.axhline(0, color='k', lw=0.8); ax.set_title(m); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/part2.png', dpi=150); plt.close()

# ---------- 3) effect of holdout split ratio & k ----------
print('\n### 3) Effect of split ratio (holdout) and k (k-fold) ###')
fracs, ks = [0.1, 0.3, 0.5, 0.7, 0.9], [2, 5, 10, 20]


def sweep(model, kind, values, n=20, sigma=0.3, reps=1000):
    data = [gen_data(n, sigma) for _ in range(reps)]
    truth = np.array([true_eout(model, X, y, sigma) for X, y in data])
    out = []
    for v in values:
        est = np.array([holdout(model, X, y, v) if kind == 'frac' else kfold(model, X, y, v)
                         for X, y in data])
        out.append(((est - truth).mean(), est.var()))
    return np.array(out)


fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for row, m in enumerate(MODELS):
    hb, hv = sweep(m, 'frac', fracs).T
    kb, kv = sweep(m, 'k', ks).T
    print(f'\n{m} holdout (frac,bias,var):', list(zip(fracs, hb.round(3).tolist(), hv.round(3).tolist())))
    print(f'{m} kfold  (k,bias,var):   ', list(zip(ks, kb.round(3).tolist(), kv.round(3).tolist())))
    for ax, x, y, title, logy in [
        (axes[row, 0], fracs, hb, f'{m}: Holdout bias vs frac', False),
        (axes[row, 1], fracs, hv, f'{m}: Holdout var vs frac', True),
        (axes[row, 2], ks, kb, f'{m}: K-fold bias vs k', False),
        (axes[row, 3], ks, kv, f'{m}: K-fold var vs k', False)]:
        ax.plot(x, y, 'o-'); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
        if logy: ax.set_yscale('log')
        if not logy: ax.axhline(0, color='gray', lw=0.6)
plt.tight_layout(); plt.savefig(f'{OUT}/part3.png', dpi=150); plt.close()

# ---------- 4) effect of n and sigma ----------
print('\n### 4) Effect of n and sigma on bias/variance ###')
n_list, sigma_list = [5, 10, 20, 50, 100], [0.0, 0.2, 0.4, 0.6, 0.8]

fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for row, m in enumerate(MODELS):
    bv_n = [bias_var(run(m, n, 0.3, reps=800)) for n in n_list]
    bv_s = [bias_var(run(m, 20, s, reps=800)) for s in sigma_list]
    print(f'\n{m} vs n (sigma=0.3):')
    print(pd.concat({n: bv['bias'] for n, bv in zip(n_list, bv_n)}, axis=1).round(4))
    print(f'\n{m} vs sigma (n=20):')
    print(pd.concat({s: bv['bias'] for s, bv in zip(sigma_list, bv_s)}, axis=1).round(4))
    for ax, x, key, ylab, title in [
        (axes[row, 0], n_list, ('bias', bv_n), 'bias', f'{m}: Bias vs n'),
        (axes[row, 1], n_list, ('var', bv_n), 'var', f'{m}: Variance vs n'),
        (axes[row, 2], sigma_list, ('bias', bv_s), 'bias', f'{m}: Bias vs sigma'),
        (axes[row, 3], sigma_list, ('var', bv_s), 'var', f'{m}: Variance vs sigma')]:
        col, bvs = key
        for method in ['Resub', 'Holdout', 'KFold']:
            ax.plot(x, [bv.loc[method, col] for bv in bvs], 'o-', label=method)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if col == 'bias': ax.axhline(0, color='gray', lw=0.6)
plt.tight_layout(); plt.savefig(f'{OUT}/part4.png', dpi=150); plt.close()

print('\nDone! plots saved in plots/')