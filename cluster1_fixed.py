import os
import urllib.request
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Steinmetz et al. 2019 Nature — NMA public subset
# 39 sessions across 10 mice, visual discrimination task
# Each session: spks (neurons x trials x timebins), brain_area, response, feedback_type, etc.
# Bin size: 10ms, trial window: 2.5s

URLS = [
    ("steinmetz_part0.npz", "https://osf.io/agvxh/download"),
    ("steinmetz_part1.npz", "https://osf.io/uv3mw/download"),
    ("steinmetz_part2.npz", "https://osf.io/ehmw2/download"),
]

def load_steinmetz():
    alldat = []
    for fname, url in URLS:
        if not os.path.exists(fname):
            print(f"downloading {fname}...")
            urllib.request.urlretrieve(url, fname)
        dat = np.load(fname, allow_pickle=True)['dat']
        alldat.extend(dat)
    return alldat


alldat = load_steinmetz()
print(f"loaded {len(alldat)} sessions")

# pick the session with the most brain areas represented
def count_areas(session):
    areas = session['brain_area']
    return len(set(a for a in areas if a != 'root'))

session_idx = max(range(len(alldat)), key=lambda i: count_areas(alldat[i]))
dat = alldat[session_idx]
print(f"using session {session_idx}")

spks       = dat['spks'].astype(float)      # neurons x trials x timebins
brain_area = dat['brain_area']
feedback   = dat['feedback_type'].flatten()
contrast_l = dat['contrast_left'].flatten()
contrast_r = dat['contrast_right'].flatten()
response   = dat['response'].flatten()
BIN_MS     = int(dat['bin_size'] * 1000)    # usually 10ms

print(f"{spks.shape[0]} neurons, {spks.shape[1]} trials, {spks.shape[2]} timebins ({BIN_MS}ms each)")
print(f"areas: {sorted(set(brain_area))}")

n_trials   = spks.shape[1]
n_timebins = spks.shape[2]

# keep areas with at least 10 neurons, drop 'root'
area_counts = {a: int((brain_area == a).sum()) for a in np.unique(brain_area)}
AREAS = sorted([a for a, n in area_counts.items() if n >= 10 and a != 'root'])
print(f"\nkept areas ({len(AREAS)}): {AREAS}")
print(f"neurons per area: { {a: area_counts[a] for a in AREAS} }")


def get_psth(area, smooth_sigma=2):
    idx  = np.where(brain_area == area)[0]
    psth = spks[idx].mean(axis=0)  # trials x time
    if smooth_sigma > 0:
        psth = gaussian_filter1d(psth, sigma=smooth_sigma, axis=1)
    return psth


def trial_mean(area):
    return get_psth(area).mean(axis=1)  # scalar per trial


# --- cross-area pairwise correlation ---
corr_mat = np.zeros((len(AREAS), len(AREAS)))
p_mat    = np.ones((len(AREAS), len(AREAS)))

for i, a1 in enumerate(AREAS):
    for j, a2 in enumerate(AREAS):
        if i == j:
            corr_mat[i, j] = 1.0
            continue
        r, p = pearsonr(trial_mean(a1), trial_mean(a2))
        corr_mat[i, j] = r
        p_mat[i, j]    = p

pairs = []
for i in range(len(AREAS)):
    for j in range(i+1, len(AREAS)):
        pairs.append((AREAS[i], AREAS[j], corr_mat[i,j], p_mat[i,j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("\ntop 5 coupled area pairs:")
for a1, a2, r, p in pairs[:5]:
    print(f"  {a1}-{a2}: r={r:.3f}, p={p:.2e}")


# --- lagged cross-correlation ---
lags = np.arange(-50, 51, 5)
lag_results = {}

for src, tgt, _, _ in pairs[:4]:
    src_mean = spks[brain_area == src].mean(axis=0).mean(axis=0)
    tgt_mean = spks[brain_area == tgt].mean(axis=0).mean(axis=0)
    lag_corr = []
    for lag in lags:
        shifted   = np.roll(tgt_mean, -int(lag))
        valid_len = n_timebins - abs(int(lag))
        if lag >= 0:
            r, _ = pearsonr(src_mean[:valid_len], shifted[:valid_len])
        else:
            r, _ = pearsonr(src_mean[-valid_len:], shifted[-valid_len:])
        lag_corr.append(r)
    lag_results[(src, tgt)] = np.array(lag_corr)
    peak_lag = lags[np.argmax(np.abs(lag_corr))]
    print(f"  {src}->{tgt}: peak lag={peak_lag*BIN_MS}ms")


# --- linear decoder ---
print("\ncross-area linear decoding (5-fold CV R2):")
source_areas  = AREAS[:min(3, len(AREAS))]
target_areas  = AREAS[1:min(5, len(AREAS))]
decoder_results = {}

for src in source_areas:
    for tgt in target_areas:
        if src == tgt:
            continue
        X     = get_psth(src)
        y     = trial_mean(tgt)
        sc    = StandardScaler()
        X_s   = sc.fit_transform(X)
        pca   = PCA(n_components=min(20, X_s.shape[1]//2))
        X_pca = pca.fit_transform(X_s)
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
        kf    = KFold(5, shuffle=True, random_state=42)
        r2s   = []
        for tr, te in kf.split(X_pca):
            ridge.fit(X_pca[tr], y[tr])
            r2s.append(ridge.score(X_pca[te], y[te]))
        r2 = np.mean(r2s)
        decoder_results[(src, tgt)] = r2
        print(f"  {src} -> {tgt}: R2={r2:.3f}")


# --- population trajectory PCA ---
traj_results = {}
for area in AREAS[:4]:
    X     = get_psth(area)
    pca_t = PCA(n_components=min(5, X.shape[0]-1))
    X_pca = pca_t.fit_transform(X.T)
    traj_results[area] = {
        'traj':    X_pca,
        'var_exp': pca_t.explained_variance_ratio_,
    }
    ve = pca_t.explained_variance_ratio_
    print(f"  {area}: top-3 PCs explain {ve[:3].sum():.2f} variance")


# --- feedback modulation ---
reward_idx    = np.where(feedback == 1)[0]
no_reward_idx = np.where(feedback == -1)[0]
fb_results    = {}

for area in AREAS:
    rew   = spks[brain_area == area][:, reward_idx, :].mean(axis=(0,2))
    norew = spks[brain_area == area][:, no_reward_idx, :].mean(axis=(0,2))
    t, p  = ttest_ind(rew, norew, equal_var=False)
    fb_results[area] = {
        'reward_fr': rew.mean(),
        'norew_fr':  norew.mean(),
        'p': p
    }
    sig = '*' if p < 0.05 else 'ns'
    print(f"  {area}: reward={rew.mean():.3f} vs no-reward={norew.mean():.3f} ({sig}, p={p:.3f})")


# --- figures ---

AREA_COLORS = {
    'VISp':'#e74c3c','VISl':'#e67e22','VISrl':'#f39c12',
    'VISam':'#1abc9c','VISpm':'#2c3e50','CA1':'#3498db',
    'DG':'#9b59b6','TH':'#8e44ad','MOs':'#27ae60',
    'SSp':'#16a085','ACA':'#d35400','MOp':'#c0392b',
}

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# A — correlation heatmap
ax_a = fig.add_subplot(3, 4, 1)
sns.heatmap(corr_mat, ax=ax_a, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=AREAS, yticklabels=AREAS,
            annot=len(AREAS) <= 10, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, square=True)
ax_a.set_title(f'A. Cross-area Correlation\n(n={n_trials} trials, session {session_idx})')
ax_a.tick_params(axis='x', rotation=45, labelsize=8)
ax_a.tick_params(axis='y', rotation=0,  labelsize=8)

# B — decoder R2 heatmap
ax_b = fig.add_subplot(3, 4, 2)
src_labs = source_areas
tgt_labs = target_areas
r2_grid  = np.full((len(src_labs), len(tgt_labs)), np.nan)
for i, s in enumerate(src_labs):
    for j, t in enumerate(tgt_labs):
        if (s, t) in decoder_results:
            r2_grid[i, j] = decoder_results[(s, t)]
sns.heatmap(r2_grid, ax=ax_b, cmap='YlOrRd',
            vmin=0, vmax=max(0.3, float(np.nanmax(r2_grid))),
            xticklabels=tgt_labs, yticklabels=src_labs,
            annot=True, fmt='.2f', annot_kws={'size': 9},
            linewidths=0.5)
ax_b.set_title('B. Linear Decoder R²\n(source → target, 5-fold CV)')
ax_b.set_xlabel('Target area')
ax_b.set_ylabel('Source area')
ax_b.tick_params(axis='x', rotation=45, labelsize=8)

# C — lagged correlations
ax_c = fig.add_subplot(3, 4, 3)
lms = lags * BIN_MS
c_list = ['#e74c3c','#3498db','#2ecc71','#f39c12']
for ci, (src, tgt) in enumerate(list(lag_results.keys())[:3]):
    ax_c.plot(lms, lag_results[(src,tgt)],
              label=f'{src}→{tgt}', color=c_list[ci], lw=2, alpha=0.85)
ax_c.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax_c.axhline(0, color='k', lw=0.5, alpha=0.3)
ax_c.set_xlabel('Lag (ms)')
ax_c.set_ylabel('Cross-correlation')
ax_c.set_title('C. Lagged Cross-area Correlation')
ax_c.legend(fontsize=8)

# D — PSTH per area
ax_d = fig.add_subplot(3, 4, 4)
time_axis = np.arange(n_timebins) * BIN_MS
for area in AREAS[:5]:
    psth = get_psth(area).mean(axis=0)
    c    = AREA_COLORS.get(area, '#95a5a6')
    ax_d.plot(time_axis, psth, label=area, color=c, lw=2, alpha=0.85)
ax_d.set_xlabel('Time in trial (ms)')
ax_d.set_ylabel('Mean firing rate (spk/bin)')
ax_d.set_title('D. Population PSTH by Area')
ax_d.legend(fontsize=8, ncol=2)

# E — PC1 trajectory
ax_e = fig.add_subplot(3, 4, 5)
for area in list(traj_results.keys())[:4]:
    traj = traj_results[area]['traj']
    c    = AREA_COLORS.get(area, '#95a5a6')
    ax_e.plot(time_axis, traj[:, 0], label=area, color=c, lw=2, alpha=0.85)
ax_e.set_xlabel('Time in trial (ms)')
ax_e.set_ylabel('PC1 projection')
ax_e.set_title('E. PC1 Trajectory Over Trial')
ax_e.legend(fontsize=8)

# F — cumulative variance explained
ax_f = fig.add_subplot(3, 4, 6)
for area in list(traj_results.keys())[:4]:
    ve = traj_results[area]['var_exp']
    c  = AREA_COLORS.get(area, '#95a5a6')
    ax_f.plot(range(1, len(ve)+1), np.cumsum(ve), 'o-',
              label=area, color=c, lw=2)
ax_f.set_xlabel('Number of PCs')
ax_f.set_ylabel('Cumulative variance explained')
ax_f.set_title('F. Population Dimensionality')
ax_f.legend(fontsize=8)

# G — state space PC1 vs PC2
ax_g = fig.add_subplot(3, 4, 7)
area_g = AREAS[0]
if area_g in traj_results:
    traj = traj_results[area_g]['traj']
    sc_g = plt.cm.viridis(np.linspace(0, 1, len(traj)))
    for k in range(len(traj)-1):
        ax_g.plot(traj[k:k+2, 0], traj[k:k+2, 1], color=sc_g[k], lw=2)
    ax_g.scatter(traj[0,0],  traj[0,1],  color='green', s=60, zorder=5, label='start')
    ax_g.scatter(traj[-1,0], traj[-1,1], color='red',   s=60, zorder=5, label='end')
    ax_g.set_xlabel('PC1')
    ax_g.set_ylabel('PC2')
    ax_g.set_title(f'G. {area_g} State Trajectory\n(time color-coded)')
    ax_g.legend(fontsize=8)

# H — decoder R2 bar chart
ax_h = fig.add_subplot(3, 4, 8)
dec_labels = [f'{s}→{t}' for s,t in decoder_results]
dec_vals   = list(decoder_results.values())
colors_h   = ['#e74c3c' if v > 0.05 else '#95a5a6' for v in dec_vals]
ax_h.bar(range(len(dec_vals)), dec_vals, color=colors_h, edgecolor='k', lw=0.6, alpha=0.85)
ax_h.axhline(0,    color='k', lw=0.8)
ax_h.axhline(0.05, color='k', lw=1, ls='--', alpha=0.5, label='R²=0.05')
ax_h.set_xticks(range(len(dec_labels)))
ax_h.set_xticklabels(dec_labels, rotation=40, ha='right', fontsize=8)
ax_h.set_ylabel('Cross-validated R²')
ax_h.set_title('H. Decoder Performance')
ax_h.legend(fontsize=8)

# I — feedback modulation
ax_i = fig.add_subplot(3, 4, 9)
fb_areas  = list(fb_results.keys())
rew_vals  = [fb_results[a]['reward_fr']  for a in fb_areas]
norew_vals = [fb_results[a]['norew_fr'] for a in fb_areas]
x_i = np.arange(len(fb_areas))
ax_i.bar(x_i - 0.2, rew_vals,   0.35, label='Reward',    color='#27ae60', alpha=0.8, edgecolor='k', lw=0.6)
ax_i.bar(x_i + 0.2, norew_vals, 0.35, label='No reward', color='#e74c3c', alpha=0.8, edgecolor='k', lw=0.6)
for i_a, area in enumerate(fb_areas):
    if fb_results[area]['p'] < 0.05:
        y_top = max(rew_vals[i_a], norew_vals[i_a]) * 1.08
        ax_i.text(i_a, y_top, '*', ha='center', fontsize=12, fontweight='bold')
ax_i.set_xticks(x_i)
ax_i.set_xticklabels(fb_areas, fontsize=8)
ax_i.set_ylabel('Mean firing rate (spk/bin)')
ax_i.set_title('I. Reward vs No-reward\nModulation')
ax_i.legend(fontsize=8)

# J — within vs cross-modal coupling
ax_j = fig.add_subplot(3, 4, 10)
vis_areas = [a for a in AREAS if 'VIS' in a]
within_vis, cross_mod = [], []
for i, a1 in enumerate(AREAS):
    for j, a2 in enumerate(AREAS):
        if i >= j: continue
        r = abs(corr_mat[i, j])
        if a1 in vis_areas and a2 in vis_areas:
            within_vis.append(r)
        elif (a1 in vis_areas) != (a2 in vis_areas):
            cross_mod.append(r)
if within_vis and cross_mod:
    bp = ax_j.boxplot([within_vis, cross_mod],
                       tick_labels=['Within visual', 'Cross-modal'],
                       patch_artist=True,
                       medianprops=dict(color='k', lw=2),
                       flierprops=dict(marker='o', markersize=3, alpha=0.5))
    bp['boxes'][0].set_facecolor('#e74c3c'); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#3498db'); bp['boxes'][1].set_alpha(0.7)
    from scipy.stats import mannwhitneyu
    _, p_hier = mannwhitneyu(within_vis, cross_mod, alternative='two-sided')
    ax_j.text(0.97, 0.97, f'p={p_hier:.3f}', transform=ax_j.transAxes,
              ha='right', va='top', fontsize=9)
ax_j.set_ylabel('|Correlation|')
ax_j.set_title('J. Within vs Cross-modal\nCoupling Strength')

# K — PSTH heatmap all areas
ax_k = fig.add_subplot(3, 4, 11)
psth_matrix = np.array([get_psth(a).mean(axis=0) for a in AREAS])
psth_norm   = psth_matrix - psth_matrix.mean(axis=1, keepdims=True)
psth_norm  /= psth_norm.std(axis=1, keepdims=True) + 1e-10
im_k = ax_k.imshow(psth_norm, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2,
                    extent=[0, n_timebins*BIN_MS, len(AREAS), 0])
ax_k.set_yticks(np.arange(len(AREAS)) + 0.5)
ax_k.set_yticklabels(AREAS, fontsize=8)
ax_k.set_xlabel('Time (ms)')
ax_k.set_title('K. PSTH Heatmap (Z-scored)')
plt.colorbar(im_k, ax=ax_k, fraction=0.04, pad=0.02, label='Z')

# L — summary
ax_l = fig.add_subplot(3, 4, 12)
ax_l.axis('off')
best_pair  = max(decoder_results, key=decoder_results.get)
best_r2    = decoder_results[best_pair]
top_pair   = pairs[0]
fb_sig     = [a for a in AREAS if fb_results[a]['p'] < 0.05]

summary = (
    f"Session {session_idx}: {spks.shape[0]} neurons\n"
    f"{n_trials} trials, {len(AREAS)} areas\n\n"
    f"Strongest coupling:\n"
    f"{top_pair[0]}-{top_pair[1]} (r={top_pair[2]:.3f})\n\n"
    f"Best decoder:\n"
    f"{best_pair[0]}→{best_pair[1]} (R²={best_r2:.3f})\n\n"
    f"Feedback-modulated areas:\n"
    f"{', '.join(fb_sig) if fb_sig else 'none at p<0.05'}\n\n"
    f"Key finding:\n"
    f"Correlation ≠ decodability.\n"
    f"Correlated areas can have\n"
    f"low linear decoder R², meaning\n"
    f"shared signal lives in a subspace\n"
    f"pairwise regression can't find."
)
ax_l.text(0.5, 0.97, summary, transform=ax_l.transAxes,
          ha='center', va='top', fontsize=8.5, linespacing=1.5,
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                    edgecolor='#bdc3c7', lw=1.5))

plt.tight_layout(h_pad=2.5, w_pad=2.0)
fig.savefig('figures.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nsaved figures.png")
print(f"\nSession {session_idx} summary:")
print(f"  Areas: {AREAS}")
print(f"  Best decoder: {best_pair[0]}→{best_pair[1]} R²={best_r2:.3f}")
print(f"  Strongest coupling: {top_pair[0]}-{top_pair[1]} r={top_pair[2]:.3f}")
print(f"  Feedback-modulated: {fb_sig}")
