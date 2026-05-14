"""
CVSC animation: two covariance ellipses, sweeping candidate directions,
locking onto the SVD solution as the maximiser of u^T rootA rootB v.

Scenes (all in one):
  1. Scatter data for A and B fades in
  2. Ellipses grow from origin
  3. Scatter fades, clean ellipses remain
  4. Candidate u (blue) sweeps; live readout of objective ticks
  5. Both u and v sweep simultaneously, objective updates
  6. Snap to SVD solution u1, v1 — objective peaks, annotated sigma_1
"""

from manim import *
import numpy as np

# ── ground truth geometry ─────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
THETA = np.radians(35)
SHARED = np.array([np.cos(THETA), np.sin(THETA)])
PERP = np.array([-np.sin(THETA), np.cos(THETA)])

COV_A = 2.8 * np.outer(SHARED, SHARED) + 0.35 * np.outer(PERP, PERP)
COV_B = 2.0 * np.outer(SHARED, SHARED) + 1.0 * np.outer(PERP, PERP)


def mat_sqrt(C):
    vals, vecs = np.linalg.eigh(C)
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T


ROOT_A = mat_sqrt(COV_A)
ROOT_B = mat_sqrt(COV_B)
M = ROOT_A @ ROOT_B
U_svd, S_svd, Vt_svd = np.linalg.svd(M)
U1 = U_svd[:, 0]
V1 = Vt_svd[0]


# objective value for any unit u, v
def objective(u_angle, v_angle):
    u = np.array([np.cos(u_angle), np.sin(u_angle)])
    v = np.array([np.cos(v_angle), np.sin(v_angle)])
    return float(u @ M @ v)


# ── helpers ───────────────────────────────────────────────────────────────────
C_A = ManimColor("#4A9EDB")
C_B = ManimColor("#3DBF8A")
C_U = ManimColor("#F5A623")
C_V = ManimColor("#E06ADB")
C_GOLD = ManimColor("#FFD700")

SCALE = 1.8  # data-units → manim units


def cov_to_manim_ellipse(cov, color, fill_opacity=0.08, stroke_width=2.5):
    vals, vecs = np.linalg.eigh(cov)
    a, b = np.sqrt(vals[1]), np.sqrt(vals[0])  # semi-axes
    angle = np.arctan2(vecs[1, 1], vecs[0, 1])  # rotation of major axis
    e = Ellipse(width=2 * a * SCALE, height=2 * b * SCALE, color=color, fill_opacity=fill_opacity, stroke_width=stroke_width)
    e.rotate(angle)
    return e


def unit_arrow(angle_rad, color, length=1.5 * SCALE, stroke_width=5):
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
    arr = Arrow(ORIGIN, direction * length, buff=0, color=color, stroke_width=stroke_width, max_tip_length_to_length_ratio=0.18)
    return arr


def scatter_dots(cov, n, color, alpha=0.35):
    pts = RNG.multivariate_normal([0, 0], cov, n)
    dots = VGroup(*[Dot(point=[p[0] * SCALE, p[1] * SCALE, 0], radius=0.025, color=color, fill_opacity=alpha) for p in pts])
    return dots


# ── main scene ────────────────────────────────────────────────────────────────
class CVSCScene(Scene):
    def construct(self):
        # ── title ─────────────────────────────────────────────────────────────
        title = Text("Cross-Validated Shared Covariance", font_size=28, color=WHITE)
        subtitle = Text("finding directions jointly large in both datasets", font_size=18, color=GRAY).next_to(title, DOWN, buff=0.2)
        title_grp = VGroup(title, subtitle).to_edge(UP, buff=0.3)
        self.play(FadeIn(title_grp), run_time=0.8)
        self.wait(0.3)

        # ── axes label ────────────────────────────────────────────────────────
        label_A = Text("dataset  A", font_size=18, color=C_A).to_corner(UL, buff=0.7).shift(DOWN * 1.0)
        label_B = Text("dataset  B", font_size=18, color=C_B).next_to(label_A, DOWN, buff=0.15)
        self.play(FadeIn(label_A), FadeIn(label_B), run_time=0.5)

        # ── 1. scatter data ───────────────────────────────────────────────────
        dots_A = scatter_dots(COV_A, 350, C_A, alpha=0.4)
        dots_B = scatter_dots(COV_B, 350, C_B, alpha=0.4)
        self.play(FadeIn(dots_A), FadeIn(dots_B), run_time=1.2)
        self.wait(0.4)

        # ── 2. fit ellipses ───────────────────────────────────────────────────
        ell_A = cov_to_manim_ellipse(COV_A, C_A, fill_opacity=0.10)
        ell_B = cov_to_manim_ellipse(COV_B, C_B, fill_opacity=0.10)
        self.play(GrowFromCenter(ell_A), GrowFromCenter(ell_B), run_time=1.0)
        self.wait(0.3)

        # ── 3. fade scatter ───────────────────────────────────────────────────
        cov_label = MathTex(r"\Sigma_A,\ \Sigma_B", font_size=28, color=GRAY)
        cov_label.next_to(ell_A, RIGHT, buff=0.4).shift(UP * 0.5)
        self.play(
            FadeOut(dots_A),
            FadeOut(dots_B),
            ell_A.animate.set_fill(opacity=0.12),
            ell_B.animate.set_fill(opacity=0.12),
            FadeIn(cov_label),
            run_time=0.8,
        )
        self.wait(0.4)

        # ── 4. introduce the objective ────────────────────────────────────────
        obj_tex = MathTex(r"\text{maximise}\quad u^\top \sqrt{\Sigma_A}\,\sqrt{\Sigma_B}\, v", font_size=24, color=YELLOW_B)
        obj_tex.to_edge(DOWN, buff=0.5)
        self.play(Write(obj_tex), run_time=1.0)
        self.wait(0.5)

        # ── 5. sweep u only, live objective counter ───────────────────────────
        u_angle_tracker = ValueTracker(np.radians(10))
        v_angle_fixed = np.arctan2(V1[1], V1[0])

        u_arrow = always_redraw(lambda: unit_arrow(u_angle_tracker.get_value(), C_U, length=1.4 * SCALE))
        v_arrow = unit_arrow(v_angle_fixed, C_V, length=1.4 * SCALE)

        u_lbl = always_redraw(lambda: MathTex("u", font_size=22, color=C_U).next_to(u_arrow.get_end(), UP + RIGHT, buff=0.1))
        v_lbl = MathTex("v", font_size=22, color=C_V).next_to(v_arrow.get_end(), UP + RIGHT, buff=0.1)

        obj_val = always_redraw(
            lambda: DecimalNumber(
                objective(u_angle_tracker.get_value(), v_angle_fixed),
                num_decimal_places=3,
                font_size=26,
                color=YELLOW_B,
            ).next_to(obj_tex, RIGHT, buff=0.3)
        )

        self.play(FadeIn(u_arrow), FadeIn(v_arrow), FadeIn(u_lbl), FadeIn(v_lbl), FadeIn(obj_val), run_time=0.6)
        self.wait(0.2)

        # sweep u
        self.play(u_angle_tracker.animate.set_value(np.radians(360 + 10)), run_time=3.5, rate_func=linear)
        self.wait(0.3)

        # ── 6. now sweep both u and v together ───────────────────────────────
        v_angle_tracker = ValueTracker(v_angle_fixed)
        v_arrow_dyn = always_redraw(lambda: unit_arrow(v_angle_tracker.get_value(), C_V, length=1.4 * SCALE))
        v_lbl_dyn = always_redraw(lambda: MathTex("v", font_size=22, color=C_V).next_to(v_arrow_dyn.get_end(), UP + RIGHT, buff=0.1))
        obj_val_both = always_redraw(
            lambda: DecimalNumber(
                objective(u_angle_tracker.get_value(), v_angle_tracker.get_value()),
                num_decimal_places=3,
                font_size=26,
                color=YELLOW_B,
            ).next_to(obj_tex, RIGHT, buff=0.3)
        )

        self.remove(v_arrow, v_lbl, obj_val)
        self.add(v_arrow_dyn, v_lbl_dyn, obj_val_both)

        # reset u, sweep both at offset speeds
        u_angle_tracker.set_value(np.radians(10))
        self.play(
            u_angle_tracker.animate.set_value(np.radians(360 + 10)),
            v_angle_tracker.animate.set_value(np.radians(360 + np.degrees(v_angle_fixed))),
            run_time=4.0,
            rate_func=linear,
        )
        self.wait(0.4)

        # ── 7. snap to SVD solution ───────────────────────────────────────────
        u_opt_angle = np.arctan2(U1[1], U1[0])
        v_opt_angle = np.arctan2(V1[1], V1[0])

        snap_note = Text("SVD finds the global maximum", font_size=20, color=C_GOLD)
        snap_note.next_to(obj_tex, UP, buff=0.25)

        self.play(
            u_angle_tracker.animate.set_value(u_opt_angle),
            v_angle_tracker.animate.set_value(v_opt_angle),
            run_time=1.2,
            rate_func=smooth,
        )
        self.play(FadeIn(snap_note), run_time=0.5)

        # label optimal arrows
        u_star = MathTex(r"u_1^*", font_size=24, color=C_U)
        v_star = MathTex(r"v_1^*", font_size=24, color=C_V)
        u_star.next_to(unit_arrow(u_opt_angle, C_U).get_end(), UP + RIGHT, buff=0.1)
        v_star.next_to(unit_arrow(v_opt_angle, C_V).get_end(), UP + RIGHT, buff=0.1)

        sigma_box = MathTex(r"\sigma_1 = " + f"{S_svd[0]:.2f}", font_size=28, color=C_GOLD)
        sigma_box.next_to(snap_note, RIGHT, buff=0.5)

        self.play(
            Write(u_star),
            Write(v_star),
            Write(sigma_box),
            run_time=0.8,
        )
        self.wait(2.0)

        # ── 8. fade to final clean frame ──────────────────────────────────────
        interpretation = Text(
            "σ₁ = shared variance explained by the top joint mode",
            font_size=18,
            color=GRAY,
        ).to_edge(DOWN, buff=0.25)
        self.play(
            FadeOut(obj_tex),
            FadeOut(obj_val_both),
            FadeOut(cov_label),
            FadeIn(interpretation),
            run_time=0.8,
        )
        self.wait(2.5)
