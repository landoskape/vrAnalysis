"""Single-reference PF projections, including paired U/V bases, for inspecting the spectrum floor."""

from dataclasses import dataclass

import numpy as np

from ...configs.cvpca import CVPCAConfig
from ...configs.stimspace import (
    StimSpaceSpectraConfig, StimspaceSVCAConfig, _cvpca_project, _cvpca_spatial_basis,
    _cvsvd_basis, _fit_svca, _to_g,
)
from ...registry import PopulationRegistry
from ..panels import FigureViewer


@dataclass(frozen=True)
class ProjectionPair:
    """One scatter's paired (points, dimensions) arrays and explicit fold/basis roles."""

    x: np.ndarray
    y: np.ndarray
    label: str
    xlabel: str
    ylabel: str


@dataclass(frozen=True)
class PlacefieldProjectionFit:
    """One reference fit, its separate projection pairs, and optional paired V basis."""

    basis: np.ndarray
    projections: dict[int, np.ndarray]
    pairs: tuple[ProjectionPair, ...]
    description: str
    right_basis: np.ndarray | None = None
    position_maps: dict[int, dict[str, np.ndarray]] | None = None


def _fit_projections(dim_method, config, prepared, reference_fold, partner_fold=2):
    """Use the spectrum's own fit/transform code; never combine reference fits."""
    if dim_method == "ss_cv":
        # Exactly _cvsvd(train[i], test[j], test[k]): i and k fit U/V, j is the
        # held-out U-side test, and k is reused on V. Points are matched neurons.
        if partner_fold == reference_fold:
            raise ValueError("ss_cv requires distinct reference_fold and partner_fold.")
        partner = _to_g(prepared["test"][partner_fold])
        u, v = _cvsvd_basis(_to_g(prepared["train"][reference_fold]), partner)
        y = partner.numpy() @ v
        projections = {
            fold: _to_g(pf).numpy() @ u
            for fold, pf in enumerate(prepared["test"])
            if fold not in (reference_fold, partner_fold)
        }
        pairs = tuple(
            ProjectionPair(
                x, y, f"Fold {fold + 1} U vs {partner_fold + 1} V", "Test fold projections onto U",
                f"Fold {partner_fold + 1} onto V (used in fit)",
            )
            for fold, x in projections.items()
        )
        return PlacefieldProjectionFit(
            u, projections, pairs,
            f"Fit: folds {reference_fold + 1} / {partner_fold + 1} (U / V); points: neurons", v,
        )
    if dim_method == "SVCA":
        model = _fit_svca(*prepared["train"][reference_fold], config.centered, prepared["num_components"])
        projections, pairs = {}, []
        for fold, (source, target) in enumerate(prepared["test"]):
            if fold == reference_fold:
                continue
            x, y = (p.T.numpy() for p in model.transform(source, target))
            projections[fold] = x
            pairs.append(ProjectionPair(
                x, y, f"Fold {fold + 1}: source U vs target V",
                "Source neurons projected onto U", "Target neurons projected onto V",
            ))
        return PlacefieldProjectionFit(
            model.get_components().numpy(), projections, tuple(pairs),
            f"Fit: temporal fold {reference_fold + 1}; test: other fold; points: spatial bins",
            model.get_components(left=False).numpy(),
            {fold: prepared["test_positions"][fold] for fold in projections} if "test_positions" in prepared else None,
        )
    if dim_method == "ss_cvpca":
        basis = _cvpca_spatial_basis(prepared["train"][reference_fold])
        projections = {
            fold: _cvpca_project(pf, basis).numpy()
            for fold, pf in enumerate(prepared["test"])
            if fold != reference_fold
        }
    elif dim_method == "reg_covariances_fixed":
        model = config.fit_fixed_model(
            prepared["torch_pfs"][reference_fold], prepared["dist_centers"], prepared["full_dist_centers"]
        )
        basis = model.pca.get_components()
        # Spatial CVPCA receives (positions, neurons). Its own PCA.transform applies the
        # fitted training mean; use it directly rather than recreating the centering here.
        projections = {
            fold: model.pca.transform(pf.T).T.numpy()
            for fold, pf in enumerate(prepared["torch_pfs"])
            if fold != reference_fold
        }
    else:
        raise ValueError(f"Unknown dim method {dim_method!r}.")
    first, *others = sorted(projections)
    pairs = tuple(
        ProjectionPair(
            projections[first], projections[fold], f"Fold {first + 1} vs {fold + 1}",
            f"Fold {first + 1} projection (held out)", "Other held-out fold projections",
        )
        for fold in others
    )
    return PlacefieldProjectionFit(
        basis.numpy(), projections, pairs, f"Reference fold {reference_fold + 1}; points: neurons",
    )


def _pearson_r(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


class PlacefieldProjectionViewer(FigureViewer):
    """Compare PF projections along one dimension from one reference fit.

    ``sessions`` accepts a sequence of B2Session objects or a ResultsAggregator. The
    mouse and example_session widgets select among them (session labels are session_uid).
    ``dim_method`` is shown as "dim method" and offers ss_cvpca, reg_covariances_fixed,
    ss_cv and SVCA (the placefield ss estimator from StimspaceSVCAConfig).
    The environment dropdown uses actual session environment IDs and restricts placefields
    before fitting for every method. It defaults to the session's first environment.
    Supply the concrete spectrum configs to reproduce the desired
    scaling, smoothing, reliability filters, bins and data splits; otherwise their original
    defaults are used. CVPCA must use spatial eigenvectors.

    All index controls are zero-based; figure fold labels are one-based. For the two PCA
    methods, the first held-out fold supplies x and the others supply y (points: neurons).
    For ss_cv, reference_fold supplies the U fit data, partner_fold supplies both the V
    fit and V test data, and each remaining fold supplies a U test scatter (points: neurons).
    partner_fold is ignored for the other methods. For SVCA, U/V act on source/target
    neuron groups in the SAME held-out temporal fold (points: spatial bins). The fit uses
    the other temporal fold. Every pair is separate, with Pearson r in the legend.

    The right panel plots SVCA's source-U and target-V scores over position, separately
    for every held-out fold and environment. The other methods contract the spatial axis
    into one score per ROI; their right panel explains why a positional profile is unavailable.

    Prepared folds and single-reference fits are cached in this object. Dimension/style
    changes do not reload data or refit, and returning to a prior reference reuses its fit.
    No averaging across bases, dimensions or test folds is performed. Call clear_cache()
    after changing underlying data. The cache is not written to disk.

    Example::

        viewer = PlacefieldProjectionViewer(
            results, mouse="ATL022", example_session=results.session_ids[0],
            dim_method="ss_cvpca", projection_dim=40,
            spectra_config=StimSpaceSpectraConfig(smooth_widths=(5.0, None)),
        )
        viewer.show()
    """

    def __init__(
        self,
        sessions,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        environment: int | None = None,
        dim_method: str = "ss_cvpca",
        projection_dim: int = 40,
        reference_fold: int = 0,
        partner_fold: int = 2,
        spectra_config: StimSpaceSpectraConfig | None = None,
        cvpca_config: CVPCAConfig | None = None,
        svca_config: StimspaceSVCAConfig | None = None,
        markersize: float = 8.0,
        alpha: float = 0.5,
        fontsize: float = 10.0,
        figsize: tuple[float, float] = (11.0, 4.5),
    ):
        sessions = list(getattr(sessions, "sessions", sessions))
        if not sessions:
            raise ValueError("Provide at least one example session.")
        self._sessions = {session.session_uid: session for session in sessions}
        spectra_config = spectra_config or StimSpaceSpectraConfig()
        self._configs = {
            "ss_cvpca": spectra_config,
            "reg_covariances_fixed": cvpca_config or CVPCAConfig(),
            "ss_cv": spectra_config,
            "SVCA": svca_config or StimspaceSVCAConfig(),
        }
        if not self._configs["reg_covariances_fixed"].use_spatial_eigenvectors:
            raise ValueError("This viewer requires use_spatial_eigenvectors=True (one point per neuron).")
        self.figsize = figsize
        self._prepared_cache = {}
        self.projection_cache = {}
        self._registries = {}
        self._sessions_by_mouse = {}
        for session in sessions:
            self._sessions_by_mouse.setdefault(session.mouse_name, []).append(session.session_uid)
        if example_session is not None and example_session not in self._sessions:
            raise ValueError(f"Unknown example_session {example_session!r}; use a session_uid from sessions.")
        if mouse is None:
            mouse = self._sessions[example_session].mouse_name if example_session else sessions[0].mouse_name
        if mouse not in self._sessions_by_mouse:
            raise ValueError(f"Unknown mouse {mouse!r}.")
        options = self._sessions_by_mouse[mouse]
        if example_session is not None and example_session not in options:
            raise ValueError("example_session does not belong to the selected mouse.")
        self.add_selection("mouse", options=list(self._sessions_by_mouse), value=mouse)
        self.add_selection("example_session", options=options, value=example_session or options[0])
        envs = list(self._sessions[example_session or options[0]].environments)
        self.add_selection("environment", options=envs, value=environment if environment is not None else envs[0])
        self.add_selection("dim method", options=list(self._configs), value=dim_method)
        self.add_integer("reference_fold", value=reference_fold, min=0, max=self._max_reference(dim_method))
        self.add_integer("partner_fold", value=partner_fold, min=0, max=3)
        self.add_integer("projection_dim", value=projection_dim, min=0, max=max(projection_dim, 0))
        self.add_float("markersize", value=markersize, min=1.0, max=50.0)
        self.add_float("alpha", value=alpha, min=0.01, max=1.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.on_change("mouse", self.update_mouse)
        self.on_change("example_session", self.update_environment)
        self.on_change(["environment", "dim method", "reference_fold", "partner_fold"], self.refresh_data)
        self.refresh_data(self.state)

    def update_mouse(self, state):
        options = self._sessions_by_mouse[state["mouse"]]
        session = state["example_session"] if state["example_session"] in options else options[0]
        self.update_selection("example_session", options=options, value=session)
        self.update_environment({**state, "example_session": session})

    def update_environment(self, state):
        options = list(self._sessions[state["example_session"]].environments)
        environment = state["environment"] if state["environment"] in options else options[0]
        self.update_selection("environment", options=options, value=environment)
        self.refresh_data({**state, "environment": environment})

    def _prepared(self, state):
        method = state["dim method"]
        config = self._configs[method]
        key = (state["example_session"], config.key(), state["environment"])
        if key not in self._prepared_cache:
            # Each method uses its own original data configuration (even vs default).
            # Keep populations in memory without writing newly generated caches to disk.
            data_key = config.data_config.key()
            if data_key not in self._registries:
                self._registries[data_key] = PopulationRegistry(
                    registry_params=config.data_config.to_registry_params(), autosave=False,
                )
            self._prepared_cache[key] = config.process(
                self._sessions[state["example_session"]], self._registries[data_key],
                placefields_only=True, environment=state["environment"],
            )
        return key, self._prepared_cache[key]

    def _get_fit(self, state):
        key, prepared = self._prepared(state)
        method = state["dim method"]
        partner = int(state.get("partner_fold", 2)) if method == "ss_cv" else None
        fit_key = (*key, method, int(state["reference_fold"]), partner)
        if fit_key not in self.projection_cache:
            self.projection_cache[fit_key] = _fit_projections(
                method, self._configs[method], prepared, int(state["reference_fold"]), partner,
            )
        return self.projection_cache[fit_key]

    @staticmethod
    def _max_reference(method):
        return {"ss_cvpca": 3, "ss_cv": 3, "reg_covariances_fixed": 2, "SVCA": 1}[method]

    def refresh_data(self, state):
        max_reference = self._max_reference(state["dim method"])
        reference = min(int(state["reference_fold"]), max_reference)
        self.update_integer("reference_fold", value=reference, max=max_reference)
        partner = int(state["partner_fold"])
        if state["dim method"] == "ss_cv" and partner == reference:
            partner = (reference + 1) % 4
            self.update_integer("partner_fold", value=partner)
        fit = self._get_fit({**state, "reference_fold": reference, "partner_fold": partner})
        max_dim = fit.basis.shape[1] - 1
        self.update_integer("projection_dim", value=min(int(state["projection_dim"]), max_dim), max=max_dim)

    def clear_cache(self):
        """Discard prepared folds, populations and projections; refit on the next plot."""
        self._prepared_cache.clear()
        self.projection_cache.clear()
        self._registries.clear()

    def plot(self, state):
        fit = self._get_fit(state)
        dim = int(state["projection_dim"])
        fontsize = state["fontsize"]
        fig, (ax, position_ax) = self.new_subplots(1, 2, figsize=self.figsize, layout="constrained")
        for color_index, pair in enumerate(fit.pairs):
            x, y = pair.x[:, dim], pair.y[:, dim]
            valid = np.isfinite(x) & np.isfinite(y)
            r = _pearson_r(x, y)
            ax.scatter(
                x[valid], y[valid], s=state["markersize"], alpha=state["alpha"], color=f"C{color_index}",
                label=f"{pair.label}: r={r:.6g} (n={valid.sum()})", linewidths=0,
            )
        ax.axhline(0, color="0.8", linewidth=0.7, zorder=0)
        ax.axvline(0, color="0.8", linewidth=0.7, zorder=0)
        ax.set_xlabel(fit.pairs[0].xlabel, fontsize=fontsize)
        ax.set_ylabel(fit.pairs[0].ylabel, fontsize=fontsize)
        ax.set_title(
            f"{state['example_session']} | env {state['environment']}\n{state['dim method']} | dimension {dim} (0-based)"
            f"\n{fit.description}", fontsize=fontsize,
        )
        ax.tick_params(labelsize=fontsize)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        ax.xaxis.get_offset_text().set_fontsize(fontsize)
        ax.yaxis.get_offset_text().set_fontsize(fontsize)
        ax.legend(fontsize=fontsize * 0.8)
        self._plot_position(position_ax, fit, state)
        return fig

    def _plot_position(self, ax, fit, state):
        dim, fontsize = int(state["projection_dim"]), state["fontsize"]
        ax.set_title(f"Dimension {dim} over position", fontsize=fontsize)
        if state["dim method"] != "SVCA":
            ax.set_axis_off()
            nroi = fit.pairs[0].x.shape[0]
            ax.text(
                0.5, 0.5,
                f"This projection produces one value per ROI\n({nroi} ROIs × 1 selected dimension).\n\n"
                "The spatial axis was contracted during projection,\nso these scores have no position axis\n"
                "and cannot be plotted as a PC placefield.",
                transform=ax.transAxes, ha="center", va="center", fontsize=fontsize,
            )
            return

        for pair_index, (fold, pair) in enumerate(zip(sorted(fit.projections), fit.pairs)):
            metadata = fit.position_maps[fold] if fit.position_maps is not None else None
            positions = metadata["position"] if metadata is not None else np.arange(len(pair.x))
            environments = metadata["environment"] if metadata is not None else np.zeros(len(pair.x), dtype=int)
            bin_indices = metadata["bin_index"] if metadata is not None else np.arange(len(pair.x))
            for env_index, env in enumerate(dict.fromkeys(environments)):
                selected = environments == env
                x = positions[selected]
                bins = bin_indices[selected]
                # Break lines across excluded positions, preserving the original position axis.
                gaps = np.flatnonzero(np.diff(bins) != 1) + 1
                for values, group, linestyle in ((pair.x, "source U", "-"), (pair.y, "target V", "--")):
                    y = values[selected, dim]
                    ax.plot(
                        np.insert(x.astype(float), gaps, np.nan), np.insert(y, gaps, np.nan),
                        color=f"C{(pair_index + env_index) % 10}", linestyle=linestyle,
                        label=f"Fold {fold + 1}, {group}" + (f", env {env}" if metadata is not None else ""),
                    )
        ax.axhline(0, color="0.8", linewidth=0.7, zorder=0)
        ax.set_xlabel("Position (cm)" if fit.position_maps is not None else "Spatial bin (flattened)", fontsize=fontsize)
        ax.set_ylabel("Projected placefield", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.legend(fontsize=fontsize * 0.8)
