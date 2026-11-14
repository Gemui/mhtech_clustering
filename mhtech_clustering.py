"""
Case study DLBDSMLUSL01 - Task 1: Mental Health in Technology-related Jobs.
Pipeline: cleaning -> feature engineering -> encoding -> PCA -> clustering
-> 2D embeddings (PCA, MDS, LLE) -> cluster profiling.
Usage: python mhtech_clustering.py <path_to_csv> <output_dir>
"""
import sys, json, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import MDS, LocallyLinearEmbedding, trustworthiness
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.RandomState(SEED)
csv, out = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 150})

raw = pd.read_csv(csv)
res = {"n_raw": int(len(raw)), "n_cols_raw": int(raw.shape[1])}
raw.columns = [re.sub(r"\s+", " ", c.replace("\xa0", " ")).strip() for c in raw.columns]
C = list(raw.columns)

# ---------- 1. Scope: employees only (HR programme targets employed staff)
res["n_selfemployed"] = int((raw[C[0]] == 1).sum())
df = raw[raw[C[0]] == 0].copy().reset_index(drop=True)
res["n_employed"] = int(len(df))

# ---------- 2. Drop free text, geographic detail and structurally empty columns
free_text = [C[37], C[39], C[48], C[49], C[51], C[61], C[58], C[60]]
miss_emp = df.isna().mean()
sparse = [c for c in df.columns if miss_emp[c] > 0.5 and c not in free_text]
res["n_missing_total_before"] = int(df.isna().sum().sum())
res["dropped_sparse"] = len(sparse)

# ---------- 3. Feature engineering of messy columns
age = df[C[55]].astype(float)
age[(age < 18) | (age > 75)] = np.nan
res["age_outliers"] = int(age.isna().sum())
age = age.fillna(age.median())

def gender(x):
    s = str(x).strip().lower()
    if re.fullmatch(r"(male|m|man|cis male|cis man|male \(cis\)|mail|malr|dude|male.*)", s) and "fe" not in s: return "male"
    if re.search(r"fem|woman|^f$|girl|she", s): return "female"
    return "other"
g = df[C[56]].map(gender)
res["gender_counts"] = g.value_counts().to_dict()

country = df[C[59]].where(df[C[59]].map(df[C[59]].value_counts()) >= 40, "Other")
role = df[C[61]].fillna("")
roles = {"Back-end": "Back-end Developer", "Front-end": "Front-end Developer", "DevOps": "DevOps/SysAdmin",
         "Leadership": "Supervisor/Team Lead|Executive Leadership", "Support": "Support", "Design": "Designer"}
role_flags = pd.DataFrame({f"role_{k}": role.str.contains(v).astype(int) for k, v in roles.items()})
role_flags["n_roles"] = role.map(lambda s: len(s.split("|")) if s else 0)
cond = df[C[51]].fillna("")
cond_n = cond.map(lambda s: len(s.split("|")) if s else 0).astype(float)
cond_flags = pd.DataFrame({
    "dx_mood": cond.str.contains("Mood").astype(int),
    "dx_anxiety": cond.str.contains("Anxiety").astype(int),
    "dx_adhd": cond.str.contains("Attention").astype(int)})

# ---------- 4. Encoding
ordinal_maps = {
 C[1]: ["1-5","6-25","26-100","100-500","500-1000","More than 1000"],
 C[9]: ["Very difficult","Somewhat difficult","Neither easy nor difficult","Somewhat easy","Very easy"],
 C[42]: ["Not open at all","Somewhat not open","Neutral","Somewhat open","Very open"],
 C[53]: ["Never","Rarely","Sometimes","Often"],
 C[54]: ["Never","Rarely","Sometimes","Often"],
 C[62]: ["Never","Sometimes","Always"],
}
tri = {"No":0.0,"Maybe":0.5,"Yes":1.0}
tri_cols = [C[i] for i in [10,11,12,13,36,38,46,47] ]
nominal = [C[i] for i in [4,5,6,7,8,14,15,40,41,43,45]]   # previous-employer block (cols 24-35) is profile-free: excluded
nominal = [c for c in nominal if c not in sparse]
binary = [C[2], C[50], C[52]]
num = pd.DataFrame(index=df.index)
for c, order in ordinal_maps.items():
    if c in sparse: continue
    m = {v: i/(len(order)-1) for i, v in enumerate(order)}
    na = df[c].isna()
    if c == C[9]:                     # "I don't know" -> own indicator, neutral position
        dk = df[c].eq("I don't know"); num["leave_dontknow"] = dk.astype(float)
        num[c] = df[c].map(m).fillna(0.5)
    elif c in (C[53], C[54]):         # "Not applicable to me" -> 0 (no interference)
        num[c] = df[c].map(m).fillna(0.0)
    else:
        num[c] = df[c].map(m).fillna(0.5)
for c in tri_cols:
    num[c] = df[c].map(tri)
    num[c + "|unsure"] = num[c].isna().astype(float)
    num[c] = num[c].fillna(0.5)
for c in binary:
    num[c] = df[c].map({"Yes":1.0,"No":0.0}).fillna(df[c]).astype(float) if df[c].dtype==object or str(df[c].dtype).startswith('str') else df[c].astype(float)
cat = pd.get_dummies(df[nominal].fillna("Not applicable"), prefix_sep="=", dtype=float)
eng = cond_flags.copy()                      # engineered from free-text diagnosis field
cont_s = pd.DataFrame({"n_conditions": cond_n / cond_n.max()}, index=df.index)   # min-max, same range as other features
X = pd.concat([num, cat, eng, cont_s], axis=1).astype(float)   # demographics/roles kept OUT of clustering, used only to profile
X = X.loc[:, X.std() > 0]
res["n_features_encoded"] = int(X.shape[1])
res["n_missing_after"] = int(X.isna().sum().sum())
Xs = X.values

# ---------- 5. PCA
pca_full = PCA(random_state=SEED).fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n_pc = int(np.searchsorted(cum, 0.80) + 1)
res["n_pc_80"] = n_pc; res["var_pc1"] = float(pca_full.explained_variance_ratio_[0]); res["var_pc2"] = float(pca_full.explained_variance_ratio_[1])
res["cum_var_first2"] = float(cum[1]); res["n_pc_70"] = int(np.searchsorted(cum, 0.70) + 1)
res["n_pc_90"] = int(np.searchsorted(cum, 0.90) + 1)
Z = PCA(n_components=n_pc, random_state=SEED).fit_transform(Xs)

fig, ax = plt.subplots(figsize=(5.2, 3))
ax.plot(range(1, 41), cum[:40], "o-", ms=3, color="#1f4e79")
ax.axhline(0.8, ls="--", color="grey", lw=.8); ax.axvline(n_pc, ls="--", color="grey", lw=.8)
ax.set_xlabel("Number of principal components"); ax.set_ylabel("Cumulative explained variance")
ax.set_title(f"{n_pc} components retain 80% of the variance"); fig.tight_layout()
fig.savefig(f"{out}/fig1_pca_variance.png"); plt.close(fig)

# ---------- 6. Choosing k
ks = range(2, 11); rows = []
for k in ks:
    km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
    gm = GaussianMixture(k, covariance_type="diag", n_init=5, random_state=SEED).fit(Z)
    ag = AgglomerativeClustering(k, linkage="ward").fit(Z)
    rows.append(dict(k=k, inertia=km.inertia_, sil_km=silhouette_score(Z, km.labels_),
                     bic=gm.bic(Z), sil_gmm=silhouette_score(Z, gm.predict(Z)),
                     sil_ward=silhouette_score(Z, ag.labels_)))
sel = pd.DataFrame(rows); res["selection"] = sel.round(4).to_dict("records")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
axs[0].plot(sel.k, sel.inertia, "o-", color="#1f4e79"); axs[0].set_title("Elbow (k-means)"); axs[0].set_ylabel("Inertia")
axs[1].plot(sel.k, sel.sil_km, "o-", label="k-means", color="#1f4e79"); axs[1].plot(sel.k, sel.sil_ward, "s-", label="Ward", color="#c55a11")
axs[1].plot(sel.k, sel.sil_gmm, "^-", label="GMM", color="#548235"); axs[1].set_title("Silhouette"); axs[1].legend(fontsize=6)
axs[2].plot(sel.k, sel.bic, "o-", color="#548235"); axs[2].set_title("BIC (GMM)")
for a in axs: a.set_xlabel("k"); a.set_xticks(range(2, 11, 2))
fig.tight_layout(); fig.savefig(f"{out}/fig2_k_selection.png"); plt.close(fig)

K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
res["K"] = K
km = KMeans(K, n_init=50, random_state=SEED).fit(Z)
lab_raw = km.labels_
# stability: bootstrap ARI (30 resamples per k), see below
stab = {}
for kk in range(2, 7):
    ref = KMeans(kk, n_init=20, random_state=SEED).fit(Z); a_ = []
    for b in range(30):
        idx = rng.choice(len(Z), len(Z), replace=True)
        a_.append(adjusted_rand_score(ref.labels_, KMeans(kk, n_init=10, random_state=b).fit(Z[idx]).predict(Z)))
    stab[kk] = [float(np.mean(a_)), float(np.std(a_))]
res["stability_by_k"] = stab
res["stability_ari_mean"], res["stability_ari_sd"] = stab[K]
gm = GaussianMixture(K, covariance_type="diag", n_init=5, random_state=SEED).fit(Z)
ag = AgglomerativeClustering(K, linkage="ward").fit(Z)
res["ari_km_gmm"] = float(adjusted_rand_score(lab_raw, gm.predict(Z)))
res["ari_km_ward"] = float(adjusted_rand_score(lab_raw, ag.labels_))
res["sil_final"] = float(silhouette_score(Z, lab_raw))
k2 = KMeans(2, n_init=50, random_state=SEED).fit(Z).labels_
res["ari_k2_k4"] = float(adjusted_rand_score(lab_raw, k2))
res["sil_per_cluster"] = {}

# relabel clusters by size
order = pd.Series(lab_raw).value_counts().index.tolist()
remap = {o: i + 1 for i, o in enumerate(order)}
lab = pd.Series(lab_raw).map(remap).values
df["cluster"] = lab
res["sizes"] = pd.Series(lab).value_counts().sort_index().to_dict()
from sklearn.metrics import silhouette_samples
ss = silhouette_samples(Z, lab); res["sil_per_cluster"] = {int(c): float(ss[lab == c].mean()) for c in range(1, K + 1)}

# ---------- 7. Embeddings for visualisation
sub = rng.choice(len(Z), min(len(Z), 1146), replace=False)
Zp = PCA(2, random_state=SEED).fit_transform(Xs)
emb = {"PCA": Zp}
emb["MDS"] = MDS(2, n_init=2, max_iter=300, random_state=SEED, normalized_stress="auto").fit_transform(Z)
emb["LLE"] = LocallyLinearEmbedding(n_neighbors=30, n_components=2, random_state=SEED, eigen_solver="dense").fit_transform(Z)
res["trust"] = {k: float(trustworthiness(Z, v, n_neighbors=15)) for k, v in emb.items()}
cols = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000", "#c00000"]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
for a, (name, E) in zip(axs, emb.items()):
    for c in range(1, K + 1):
        m = lab == c; a.scatter(E[m, 0], E[m, 1], s=5, alpha=.6, color=cols[c-1], label=f"C{c}")
    a.set_title(f"{name} (T={res['trust'][name]:.2f})"); a.set_xticks([]); a.set_yticks([])
axs[0].legend(markerscale=2, fontsize=6, loc="best"); fig.tight_layout()
fig.savefig(f"{out}/fig3_embeddings.png"); plt.close(fig)

# ---------- 8. Profiles of clusters on interpretable indicators
def yes(c, v="Yes"): return (df[c] == v).astype(float)
prof = pd.DataFrame({
 "Age (mean)": age,
 "Female": (g == "female").astype(float),
 "Company > 500 staff": df[C[1]].isin(["500-1000", "More than 1000"]).astype(float),
 "US-based": (df[C[59]] == "United States of America").astype(float),
 "Employer offers MH benefits": yes(C[4]),
 "Knows care options": yes(C[5]),
 "Employer discussed MH formally": yes(C[6]),
 "Employer offers MH resources": yes(C[7]),
 "Anonymity protected": yes(C[8]),
 "Takes MH as seriously as physical": yes(C[14]),
 "Expects negative consequences (MH)": yes(C[10]),
 "Comfortable with supervisor": yes(C[13]),
 "Comfortable with coworkers": yes(C[12]),
 "Observed negative consequences": yes(C[15]),
 "Would raise MH in interview": yes(C[38]),
 "Family history": yes(C[45]),
 "Currently MH disorder": yes(C[47]),
 "Professional diagnosis": (df[C[50]] == "Yes").astype(float),
 "Sought treatment": df[C[52]].astype(float),
 "Interference when untreated: often": (df[C[54]] == "Often").astype(float),
 "Work remotely always": (df[C[62]] == "Always").astype(float),
})
P = prof.groupby(lab).mean().T
P.columns = [f"C{c}" for c in P.columns]
overall = prof.mean(); P.insert(0, "All", overall)
res["profile"] = P.round(3).to_dict()
res["role_leadership_by_cluster"] = role_flags["role_Leadership"].groupby(lab).mean().round(3).to_dict()
res["role_backend_by_cluster"] = role_flags["role_Back-end"].groupby(lab).mean().round(3).to_dict()
res["role_flags_all"] = role_flags.drop(columns="n_roles").mean().round(3).to_dict()
res["exp_all_gender"] = g.value_counts(normalize=True).round(3).to_dict()
P.round(2).to_csv(f"{out}/cluster_profiles.csv")

Pz = P.drop(columns="All").copy()
rel = Pz.sub(P["All"], axis=0)
rows_ = [r for r in Pz.index if r != "Age (mean)"]
rel_pct = rel.loc[rows_]
fig, ax = plt.subplots(figsize=(6.6, 5.2))
im = ax.imshow(rel_pct.values * 100, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
ax.set_xticks(range(K)); ax.set_xticklabels([f"C{c} (n={res['sizes'][c]})" for c in range(1, K + 1)])
ax.set_yticks(range(len(rows_))); ax.set_yticklabels(rows_, fontsize=7)
for i in range(len(rows_)):
    for j in range(K):
        ax.text(j, i, f"{Pz.loc[rows_[i]].iloc[j]*100:.0f}", ha="center", va="center", fontsize=6.5)
cb = fig.colorbar(im, ax=ax, fraction=.04); cb.set_label("Deviation from overall share (pp)", fontsize=7)
ax.set_title("Cluster profiles: share of members (%) per indicator", fontsize=9); fig.tight_layout()
fig.savefig(f"{out}/fig4_profile_heatmap.png"); plt.close(fig)

# ---------- 9. Which encoded features drive the clusters (one-vs-rest effect size)
Xd = X.copy(); Xd["c"] = lab
eff = {}
for c in range(1, K + 1):
    m = Xd["c"] == c
    d = (Xd[m].drop(columns="c").mean() - Xd[~m].drop(columns="c").mean())
    eff[c] = d.abs().sort_values(ascending=False).head(6).index.tolist()
res["top_drivers"] = {int(k): v for k, v in eff.items()}
# PCA loadings
pc = PCA(3, random_state=SEED).fit(Xs)
ld = pd.DataFrame(pc.components_.T, index=X.columns, columns=["PC1", "PC2", "PC3"])
res["loadings"] = {c: ld[c].abs().sort_values(ascending=False).head(6).index.tolist() for c in ld}
res["loadings_val"] = {c: ld[c].loc[ld[c].abs().sort_values(ascending=False).head(6).index].round(3).to_dict() for c in ld}
df[[C[55], "cluster"]].to_csv(f"{out}/cluster_assignment_sample.csv", index=False)
json.dump(res, open(f"{out}/results.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("profile", "loadings_val")}, indent=1, default=str))
