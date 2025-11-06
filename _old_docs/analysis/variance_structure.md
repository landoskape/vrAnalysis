# vrAnalysis Documentation: variance_structure

This is an analysis module designed to analyze variance structure in place coding for single sessions.


cvPCA Example
-------------
```python
mouse_name = 'ATL012'
ises = sessiondb.iterSessions(mouseName=mouse_name)
ses = ises[-8] # np.random.choice(ises)
print(ses.sessionPrint())
pcss = analysis.VarianceStructure(ses, distStep=1)

envnum = None # means get all environments
smooth = 0.1
spkmaps = pcss.prepare_spkmaps(envnum=envnum, smooth=smooth)
print([s.shape for s in spkmaps])

cv_by_env, cv_across = pcss.do_cvpca(spkmaps, by_trial=False)
cv_by_env_tt, cv_across_tt = pcss.do_cvpca(spkmaps, by_trial=True)

norm = lambda x: x #/ np.sum(x)
ylog = False

cmap = mpl.colormaps['rainbow'].resampled(len(cv_by_env))
figsize = 4
fig, ax = plt.subplots(2, 2, figsize=(2*figsize, 2*figsize), layout='constrained')
for ii, cc in enumerate(cv_by_env):
    ax[0, 0].plot(range(1, len(cc)+1), norm(cc), color=cmap(ii), label=f"env{ii}")
    ax[0, 1].plot(range(1, len(cc)+1), np.cumsum(norm(cc)), color=cmap(ii), label=f"env{ii}")
ax[0, 0].plot(range(1, len(cc)+1), norm(cv_across), color='k', label='all')
ax[0, 1].plot(range(1, len(cc)+1), np.cumsum(norm(cv_across)), color='k', label='all')

ax[0, 0].set_xscale('log')
ax[0, 0].set_xlabel('Dimension')
ax[0, 0].set_ylabel('Variance')
ax[0, 0].set_title('ESpectrum - Each vs. Together')
ax[0, 1].set_xscale('log')

if ylog:
    ax[0, 0].set_ylim(1e-3)
    ax[0, 0].set_yscale('log')

ax[0, 1].set_xlabel('Dimension')
ax[0, 1].set_ylabel('Cumulative Variance')
ax[0, 1].set_title('Cumulative - Each vs. Together')
ax[0, 1].legend()



for ii, cc in enumerate(cv_by_env_tt):
    ax[1, 0].plot(range(1, len(cc)+1), norm(cc), color=cmap(ii), label=f"env{ii}")
    ax[1, 1].plot(range(1, len(cc)+1), np.cumsum(norm(cc)), color=cmap(ii), label=f"env{ii}")
ax[1, 0].plot(range(1, len(cc)+1), norm(cv_across_tt), color='k', label='all')
ax[1, 1].plot(range(1, len(cc)+1), np.cumsum(norm(cv_across_tt)), color='k', label='all')

ax[1, 0].set_xscale('log')
ax[1, 0].set_xlabel('Dimension')
ax[1, 0].set_ylabel('Variance')
ax[1, 0].set_title('Without prior trial averaging')
ax[1, 1].set_xscale('log')

if ylog:
    ax[1, 0].set_ylim(1e-3)
    ax[1, 0].set_yscale('log')

ax[1, 1].set_xlabel('Dimension')
ax[1, 1].set_ylabel('Cumulative Variance')
ax[1, 1].set_title('Without prior trial averaging')
ax[1, 1].legend()
plt.show()
```


cv Fourier Example
-------------
```python
mouse_name = 'ATL012'
ises = sessiondb.iterSessions(mouseName=mouse_name)
ses = ises[-8] # np.random.choice(ises)
print(ses.sessionPrint())
pcss = analysis.VarianceStructure(ses, distStep=1)

envnum = None # means get all environments
smooth = 0.1
spkmaps = pcss.prepare_spkmaps(envnum=envnum, smooth=smooth)
print([s.shape for s in spkmaps])

freqs, cvf_by_env = pcss.do_cvfourier(spkmaps, by_trial=False)
title = ["cosine", "sine"]
cmap = mpl.colormaps['rainbow'].resampled(len(cvf_by_env))
figsize = 4

fig, ax = plt.subplots(1, 2, figsize=(2*figsize, figsize), layout='constrained')
for aa in range(2):
    for ii, cc in enumerate(cvf_by_env):
        ax[aa].plot(freqs, cc[aa], color=cmap(ii), label=f"env{ii}")
    
    ax[aa].set_xlabel('Frequency')
    ax[aa].set_ylabel('Train/Test Correlation')
    ax[aa].set_title(title[aa])
    ax[aa].legend(loc='upper right')

plt.show()
```