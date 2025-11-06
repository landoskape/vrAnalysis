import numpy as np
import matplotlib.pyplot as plt

from _old_vrAnalysis import analysis
from _old_vrAnalysis import helpers
from _old_vrAnalysis import tracking


mouse_name = "ATL022"
track = tracking.tracker(mouse_name)
pcm = analysis.placeCellMultiSession(track, autoload=False)
ises = 7
pcss = analysis.placeCellSingleSession(pcm.pcss[ises].vrexp, keep_planes=[1, 2, 3, 4], autoload=False)
split_params = dict(total_folds=2, train_folds=1)
pcss.define_train_test_split(**split_params)
pcss.load_data(new_split=False)


train_spkmaps = pcss.get_spkmap(average=True, smooth=1, trials="train")
test_spkmaps = pcss.get_spkmap(average=True, smooth=1, trials="test")

idx_nan = np.any(np.stack([np.any(np.isnan(t), axis=0) for t in train_spkmaps] + [np.any(np.isnan(t), axis=0) for t in test_spkmaps]), axis=0)
train_spkmaps = [t[:, ~idx_nan] for t in train_spkmaps]
test_spkmaps = [t[:, ~idx_nan] for t in test_spkmaps]

nc = 80
cvpca = [helpers.cvPCA(train.T, test.T, nc=80) for train, test in zip(train_spkmaps, test_spkmaps)]


cvpca_v = [helpers.smart_pca(train, centered=True)[1][:, :nc] for train in train_spkmaps]
train_proj = [v.T @ (train - train.mean(axis=1, keepdims=True)) for v, train in zip(cvpca_v, train_spkmaps)]
test_proj = [v.T @ (test - test.mean(axis=1, keepdims=True)) for v, test in zip(cvpca_v, test_spkmaps)]
print([v.shape for v in cvpca_v])


plt.close("all")


ineg = [np.where(s < 0)[0][0] for s in cvpca]
xv = range(1, nc + 1)
fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
ax[0, 0].plot(xv, cvpca[0], c="k")
ax[0, 1].plot(xv, cvpca[1], c="k")
ax[0, 0].set_xlabel("Component")
ax[0, 0].set_ylabel("C-V Variance")
ax[0, 0].axvline(ineg[0] + 1, color="r")
ax[0, 1].axvline(ineg[1] + 1, color="r")
ax[0, 0].set_yscale("log")
ax[0, 1].set_yscale("log")
ax[1, 0].plot(train_proj[0][ineg[0]], "k", label="Train")
ax[1, 0].plot(test_proj[0][ineg[0]], "b", label="Test")
ax[1, 1].plot(train_proj[1][ineg[1]], "k", label="Train")
ax[1, 1].plot(test_proj[1][ineg[1]], "b", label="Test")
ax[1, 0].set_xlabel("Train Projection onto Component")
plt.show()


check = cvpca_v[0][:, ineg[0]]
print(check.shape)
idx_big = np.where(np.abs(check) > np.percentile(np.abs(check), 95))[0]
idx_sorted = np.argsort(check)
idx_big_sorted = np.argsort(check[idx_big])

fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
ax[0].imshow(train_spkmaps[0][idx_sorted], aspect="auto", cmap="gray", vmin=0, vmax=2)
ax[1].imshow(test_spkmaps[0][idx_sorted], aspect="auto", cmap="gray", vmin=0, vmax=2)
ax[1].set_xlabel("Position")
ax[1].set_ylabel("ROIs - Sorted by beta")
plt.show()


pfidx = pcss.get_place_field(train_spkmaps[0][idx_big])[1]

fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
ax[0].imshow(train_spkmaps[0][idx_big][pfidx], aspect="auto", cmap="gray", vmin=0, vmax=2)
ax[1].imshow(test_spkmaps[0][idx_big][pfidx], aspect="auto", cmap="gray", vmin=0, vmax=2)
ax[1].set_xlabel("Position")
ax[0].set_ylabel("ROIs high weight on component")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
ax[0].imshow(train_spkmaps[0][idx_big][idx_big_sorted], aspect="auto", cmap="gray", vmin=0, vmax=2)
ax[1].imshow(test_spkmaps[0][idx_big][idx_big_sorted], aspect="auto", cmap="gray", vmin=0, vmax=2)
ax[1].set_xlabel("Position")
ax[0].set_ylabel("ROIs high weight on component (sorted)")
plt.show()


envselect = lambda tuples, idx: map(lambda x: x[idx], tuples)
relmse, relcor, relloo = envselect(pcss.get_reliability_values(), 0)

avgmse, avgcor = [], []
for idim in range(cvpca_v[0].shape[1]):
    idx_big = cvpca_v[0][:, idim] > np.percentile(cvpca_v[0][:, idim], 95)
    avgmse.append(np.mean(relmse[idx_big]))
    avgcor.append(np.mean(relcor[idx_big]))

fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
ax[0].plot(avgmse)
ax[1].plot(avgcor)
ax[0].set_xlabel("Component")
ax[0].set_ylabel("RELMSE - Top5% ROIs by Beta")
ax[1].set_xlabel("Component")
ax[1].set_ylabel("RELCOR - Top5% ROIs by Beta")

plt.show()


xpos = range(train_spkmaps[0].shape[1])
ishow = np.random.permutation(idx_big.sum())[:4]

xxtrain = train_spkmaps[0][idx_big][ishow]
xxtest = test_spkmaps[0][idx_big][ishow]
xxtrain = xxtrain / np.max(xxtrain, axis=1, keepdims=True)
xxtest = xxtest / np.max(xxtest, axis=1, keepdims=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
ax[0].plot(xpos, xxtrain.T)
ax[0].set_xlabel("Position")
ax[0].set_ylabel("Normalized Activity")
ax[0].set_title("Example ROIs - Train")
ax[1].plot(xpos, xxtest.T)
ax[0].set_xlabel("Position")
ax[0].set_ylabel("Normalized Activity")
ax[0].set_title("Example ROIs - Test")
plt.show()
