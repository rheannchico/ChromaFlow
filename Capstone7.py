import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import io
import time
import hashlib
from collections import Counter

# --------------------------- Data→Vis mappings ------------------------------

INTENSITY_TO_WEIGHT = {"Low": 1.2, "Moderate": 1.6, "High": 2.1}
TYPE_TO_STYLE = {
    "Yoga": "Floral",
    "Walking": "Floral",
    "Running": "Wave",
    "Cycling": "Wave",
    "Weightlifting": "Burst",
}

# Intensity-only colormaps
INTENSITY_CMAP = {
    "Low": plt.cm.Greens,
    "Moderate": plt.cm.Oranges,
    "High": plt.cm.Reds,
}

def cmap_for_intensity(intensity):
    return INTENSITY_CMAP.get(intensity, plt.cm.Greys)

# Extra encodings for embedding-style layout
INTENSITY_VAL = {"Low": 0.3, "Moderate": 0.65, "High": 1.0}

# Spread the workout types around the circle like "clusters"
TYPE_BASE_ANGLE = {
    "Yoga":          0.0,
    "Walking":       2 * np.pi / 5,
    "Running":       2 * 2 * np.pi / 5,
    "Cycling":       3 * 2 * np.pi / 5,
    "Weightlifting": 4 * 2 * np.pi / 5,
}

lerp = lambda a, b, t: a + (b - a) * t

def rand_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else:
        np.random.seed(None)
        random.seed()

# ---------- NEW: simple helper to make a user-specific seed -----------------

def user_profile_seed(age, gender):
    """
    Convert age + gender into a stable integer seed.
    This lets each person get a unique visual style.
    """
    key = f"{age}-{gender}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # take first 8 hex chars as an int


# ------------------------------ Titles/Captions ------------------------------

def summarize_workouts(workouts):
    total_min = sum(w["duration"] for w in workouts)
    by_type = {}
    by_intensity = {"Low": 0, "Moderate": 0, "High": 0}
    for w in workouts:
        by_type[w["type"]] = by_type.get(w["type"], 0) + 1
        by_intensity[w["intensity"]] += 1
    return total_min, by_type, by_intensity


def pick_art_title(workouts, age=None, gender=None):
    if not workouts:
        return "ChromaFlow", "Add workouts to generate art."

    total_min, by_type, by_intensity = summarize_workouts(workouts)
    has_high = by_intensity.get("High", 0) > 0

    floral_bias = (
        by_type.get("Yoga", 0) + by_type.get("Walking", 0)
        >= max(1, (by_type.get("Running", 0)
                   + by_type.get("Cycling", 0)
                   + by_type.get("Weightlifting", 0)))
    )
    wave_bias = (
        by_type.get("Running", 0) + by_type.get("Cycling", 0)
        > by_type.get("Weightlifting", 0)
    )
    burst_bias = by_type.get("Weightlifting", 0) >= max(
        by_type.get("Running", 0),
        by_type.get("Cycling", 0),
        by_type.get("Yoga", 0),
        by_type.get("Walking", 0),
    )

    if floral_bias and not has_high:
        title = "Calm Bloom"
    elif burst_bias and has_high:
        title = "Power Burst"
    elif wave_bias and not has_high:
        title = "Rhythm Flow"
    elif wave_bias and has_high:
        title = "Heat Wave"
    else:
        title = "Chroma Tapestry"

    kinds = ", ".join([k for k, v in by_type.items() if v > 0])

    profile_bits = []
    if age is not None:
        profile_bits.append(f"Age {age}")
    if gender and gender != "Prefer not to say":
        profile_bits.append(gender)

    profile_str = " • ".join(profile_bits) if profile_bits else ""
    caption_core = f"{len(workouts)} workout(s) • {total_min} min • Types: {kinds or '—'}"
    caption = f"{caption_core}{(' • ' + profile_str) if profile_str else ''}"
    return title, caption


# ------------------------------- Draw functions ------------------------------

def draw_floral(ax, duration, intensity, cmap, seed=None):
    rand_seed(seed)
    turns = max(0.2, duration / 10.0)
    petals = max(5, int(duration / 5))
    k = petals
    a = 0.98
    total_theta = 2 * np.pi * turns
    steps = max(800, int(1200 * turns))
    theta = np.linspace(0, total_theta, steps)

    r = a * np.abs(np.cos(k * theta)) ** 0.85
    r += 0.02 * np.sin(5 * theta + np.random.rand() * 2 * np.pi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    base_lw = INTENSITY_TO_WEIGHT.get(intensity, 1.4)

    for i in range(steps - 1):
        c = cmap(i / steps)
        ax.plot(
            x[i : i + 2],
            y[i : i + 2],
            color=c,
            linewidth=base_lw * lerp(0.7, 1.15, i / steps),
            alpha=lerp(0.55, 0.9, i / steps),
        )

    dot_every = max(25, int(steps / (petals * 2)))
    for i in range(0, steps, dot_every):
        ax.scatter(
            x[i],
            y[i],
            s=18,
            color=cmap(i / steps),
            alpha=0.85,
            marker="o",
        )


def draw_wave(ax, duration, intensity, cmap, seed=None):
    rand_seed(seed)
    rings_full = int(duration // 10) + 1
    frac = (duration % 10) / 10.0
    total_rings = rings_full
    base_lw = INTENSITY_TO_WEIGHT.get(intensity, 1.4)
    steps = 800

    for ri in range(total_rings):
        theta_max = (
            2 * np.pi
            if not (ri == total_rings - 1 and frac > 0 and duration >= 10)
            else 2 * np.pi * frac
        )
        theta = np.linspace(0, theta_max, steps)
        R = lerp(0.2, 1.0, (ri + 1) / total_rings)
        freq = 6 + int(ri * 0.8)
        amp = lerp(0.015, 0.09, (ri + 1) / total_rings)
        r = R + amp * np.sin(freq * theta + np.random.rand() * 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        for i in range(steps - 1):
            c = cmap((ri + i / steps) / (total_rings + 0.5))
            ax.plot(
                x[i : i + 2],
                y[i : i + 2],
                color=c,
                linewidth=base_lw,
                alpha=0.75,
            )

        dash_step = max(40, int(steps / (10 + ri * 2)))
        for i in range(0, steps, dash_step):
            j = min(i + 3, steps - 1)
            ax.plot(
                x[i:j],
                y[i:j],
                color=cmap(0.9),
                linewidth=base_lw * 1.4,
                alpha=0.5,
            )


def draw_burst(ax, duration, intensity, cmap, seed=None):
    rand_seed(seed)
    base_rays = max(18, int(duration * 1.3))
    frac = (duration % 10) / 10.0
    base_lw = INTENSITY_TO_WEIGHT.get(intensity, 1.4)
    angles = np.linspace(0, 2 * np.pi, base_rays, endpoint=False)
    np.random.shuffle(angles)

    for idx, ang in enumerate(angles):
        L = lerp(0.35, 1.05, 1 - (idx / max(1, base_rays - 1)))
        tail_scale = lerp(0.25, 1.0, frac) if idx > 0.8 * base_rays else 1.0
        jitter = np.random.uniform(-0.05, 0.05)
        ang_j = ang + jitter

        x0, y0 = 0.0, 0.0
        x1, y1 = (L * tail_scale) * np.cos(ang_j), (L * tail_scale) * np.sin(ang_j)

        t_steps = 50
        xs = np.linspace(x0, x1, t_steps)
        ys = np.linspace(y0, y1, t_steps)

        for i in range(t_steps - 1):
            c = cmap(i / t_steps)
            ax.plot(
                xs[i : i + 2],
                ys[i : i + 2],
                color=c,
                linewidth=base_lw,
                alpha=lerp(0.85, 0.45, i / t_steps),
            )

        ax.scatter(
            [x1],
            [y1],
            color=cmap(0.95),
            s=28,
            alpha=0.9,
            marker="*",
        )

    rings = 3 + int(duration / 20)
    for r_i in range(rings):
        theta = np.linspace(0, 2 * np.pi, 600)
        R = lerp(0.15, 1.15, (r_i + 1) / rings)
        ripple = 0.02 * np.sin((6 + r_i) * theta + np.random.rand() * 2 * np.pi)
        x = (R + ripple) * np.cos(theta)
        y = (R + ripple) * np.sin(theta)
        ax.plot(
            x,
            y,
            color=cmap((r_i + 1) / rings),
            linewidth=base_lw * 0.9,
            alpha=0.5,
            linestyle="--",
        )


# --------------------- Background & embedding-style layers -------------------

def draw_background_field(ax, workouts, seed=None):
    """Soft radial/noise field that reacts to max intensity present."""
    if not workouts:
        return

    rand_seed(seed)

    rank = {"Low": 0, "Moderate": 1, "High": 2}
    max_int = max((w["intensity"] for w in workouts), key=lambda x: rank[x])
    cmap = cmap_for_intensity(max_int)

    n = 260
    xs = np.linspace(-1.3, 1.3, n)
    ys = np.linspace(-1.3, 1.3, n)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y, X)

    noise = np.zeros_like(R)
    for k in [3, 6, 11]:
        noise += np.sin(
            k * angle + np.random.rand() * 2 * np.pi
        ) * np.exp(-R * (0.6 + 0.4 * np.random.rand()))

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    rgba = cmap(noise)
    alpha_mask = (1.1 - R).clip(0, 1)
    rgba[..., 3] = 0.3 * alpha_mask

    ax.imshow(
        rgba,
        extent=[-1.3, 1.3, -1.3, 1.3],
        origin="lower",
        zorder=0,
    )


# ---------------------------- Shell & hero section ---------------------------

st.set_page_config(page_title="ChromaFlow", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #3a1c71, #d76d77, #ffaf7b, #43cea2);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }
    .abstract-bg {
        position: relative;
        width: 100%;
        height: 180px;
        overflow: hidden;
        margin-bottom: -110px;
    }
    .abstract-bg span {
        position: absolute;
        border-radius: 50%;
        opacity: .25;
        animation: float 12s infinite ease-in-out;
        filter: blur(1px);
    }
    @keyframes float {
        0% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-40px) scale(1.08); }
        100% { transform: translateY(0) scale(1); }
    }
    .abstract-bg span:nth-child(1){
        background:#ffffff22; width:160px; height:160px; left:10%; top:10%; animation-delay:0s;
    }
    .abstract-bg span:nth-child(2){
        background:#ffffff33; width:120px; height:120px; left:72%; top:38%; animation-delay:2.5s;
    }
    .abstract-bg span:nth-child(3){
        background:#ffffff1a; width:220px; height:220px; left:42%; top:68%; animation-delay:5s;
    }
    .title-text {
        text-align:center; color:#fff; position:relative; z-index:2;
    }
    .title-text h1 {
        font-size:3.2em; font-weight:800; letter-spacing:.5px;
        text-shadow:0 0 18px rgba(255,255,255,.35); margin:0;
    }
    .title-text p {
        font-size:1.05em; opacity:.9; margin-top:6px; margin-bottom:0;
    }
    .intro {
        max-width:800px; margin:0 auto; color:#fff; opacity:.95;
        text-align:center; font-size:1.02em;
    }
    .art-frame {
        background:rgba(255,255,255,.08);
        border:1px solid rgba(255,255,255,.18);
        border-radius:18px;
        box-shadow:0 10px 40px rgba(0,0,0,.25);
        padding:18px;
        margin-top:8px;
    }
    .art-caption {
        text-align:center; color:#fff; margin-top:.6rem;
    }
    .gallery-h {
        color:#fff; margin-top:2rem;
    }
    </style>
    <div class="abstract-bg"><span></span><span></span><span></span></div>
    <div class="title-text">
        <h1>ChromaFlow</h1>
        <p>Transforming movement into art</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="intro">
        <p><strong>ChromaFlow</strong> turns your workouts into engaging art.</p>
        <p>In a world where health and wellness are increasingly relevant, ChromaFlow gives you a new way to celebrate your movement - not with numbers or graphs, but through color, shape, and creativity. Each workout becomes a visual story. A reminder of your energy, effort, and progress in a form that’s personal and creative.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ State & inputs -------------------------------

if "workouts" not in st.session_state:
    st.session_state.workouts = []
if "gallery" not in st.session_state:
    st.session_state.gallery = []
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None
# NEW: store profile info in session_state
if "profile_age" not in st.session_state:
    st.session_state.profile_age = 30
if "profile_gender" not in st.session_state:
    st.session_state.profile_gender = "Prefer not to say"

# NEW: Profile section
st.sidebar.header("👤 Your Profile")
age = st.sidebar.number_input("Age", min_value=13, max_value=100, value=st.session_state.profile_age, step=1)
gender = st.sidebar.selectbox(
    "Gender",
    ["Prefer not to say", "Female", "Male", "Non-binary", "Other"],
    index=["Prefer not to say", "Female", "Male", "Non-binary", "Other"].index(
        st.session_state.profile_gender
    ) if st.session_state.profile_gender in ["Prefer not to say", "Female", "Male", "Non-binary", "Other"] else 0,
)
st.session_state.profile_age = age
st.session_state.profile_gender = gender

st.sidebar.markdown("---")
st.sidebar.header("➕ Add Workout")
w_type = st.sidebar.selectbox("Type", ["Yoga", "Walking", "Running", "Cycling", "Weightlifting"])
w_duration = st.sidebar.slider("Duration (min)", 1, 180, 35, 1)
w_intensity = st.sidebar.radio("Intensity", ["Low", "Moderate", "High"], horizontal=True)

if st.sidebar.button("Add"):
    st.session_state.workouts.append(
        {"type": w_type, "duration": w_duration, "intensity": w_intensity}
    )
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Workouts"):
    st.session_state.workouts = []
    st.rerun()

# --------------------------------- Render -----------------------------------

st.subheader("Current Workouts")
if not st.session_state.workouts:
    st.info("No workouts yet. Add some on the left!")
else:
    for i, w in enumerate(st.session_state.workouts, 1):
        st.write(f"**{i}. {w['type']} — {w['duration']} min — {w['intensity']} intensity**")

styled_img_bytes = None

if st.session_state.workouts:
    bg_color = "#1a1a1a" if any(
        w["intensity"] == "High" for w in st.session_state.workouts
    ) else "#F7F7F7"

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # NEW: user-specific seed
    u_seed = user_profile_seed(st.session_state.profile_age, st.session_state.profile_gender)

    # Background field (personalized by user seed)
    draw_background_field(ax, st.session_state.workouts, seed=42 + (u_seed % 1000))

    # Motif based only on the **most recent** workout
    main = st.session_state.workouts[-1]  # last logged workout

    cmap = cmap_for_intensity(main["intensity"])
    motif = TYPE_TO_STYLE.get(main["type"], "Wave")

    # NEW: keep motif reproducible per user
    motif_seed = 12345 + (u_seed % 100000)

    if motif == "Floral":
        draw_floral(ax, main["duration"], main["intensity"], cmap, seed=motif_seed)
    elif motif == "Wave":
        draw_wave(ax, main["duration"], main["intensity"], cmap, seed=motif_seed)
    else:
        draw_burst(ax, main["duration"], main["intensity"], cmap, seed=motif_seed)

    ax.set_aspect("equal")
    ax.axis("off")

    buf_raw = io.BytesIO()
    fig.savefig(
        buf_raw,
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf_raw.seek(0)
    styled_img_bytes = buf_raw.getvalue()
    plt.close(fig)

    st.markdown('<div class="art-frame">', unsafe_allow_html=True)
    st.image(styled_img_bytes, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    title, subcap = pick_art_title(
        st.session_state.workouts,
        age=st.session_state.profile_age,
        gender=st.session_state.profile_gender,
    )
    st.markdown(
        f'<div class="art-caption"><h3 style="margin:0;">{title}</h3><div>{subcap}</div></div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "💾 Download PNG",
        data=styled_img_bytes,
        file_name=f"ChromaFlow_{title.replace(' ', '_')}.png",
        mime="image/png",
    )

# -------------------------------- Gallery -----------------------------------

st.markdown('<h2 class="gallery-h">My Gallery</h2>', unsafe_allow_html=True)

if styled_img_bytes is not None:
    img_hash = hashlib.md5(styled_img_bytes).hexdigest()
    if st.session_state.last_hash != img_hash:
        st.session_state.gallery.append(
            {"bytes": styled_img_bytes, "ts": time.time()}
        )
        st.session_state.last_hash = img_hash

if st.session_state.gallery:
    cols = st.columns(3)
    for i, item in enumerate(reversed(st.session_state.gallery)):
        with cols[i % 3]:
            st.image(item["bytes"], caption=None, use_container_width=True)

    if st.button("🧹 Clear Gallery"):
        st.session_state.gallery = []
        st.session_state.last_hash = None
        st.rerun()
else:
    st.info("Your gallery is empty. Generate artwork to start your collection ✨")
