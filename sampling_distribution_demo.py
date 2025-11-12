# sampling_distribution_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Sampling Distribution Demo", layout="centered")

st.title("🎲 Sampling Distribution of the Mean")

# --- Controls for population shape ---
shape = st.slider(
    "Population skew: Symmetric ←→ Skewed",
    min_value=0.0, max_value=1.0, value=0.0, step=0.01,
)
uniformity = st.slider(
    "Population shape: Uniform ←→ Normal",
    min_value=0.0, max_value=1.0, value=0.0, step=0.01,
    help="0 = Uniform, 1 = Normal (Gaussian)"
)
n = st.slider(
    "Sample size n",
    min_value=1, max_value=50, value=5,
    help="Each sample has n observations."
)
num_samples = st.slider(
    "Number of samples to simulate",
    min_value=100, max_value=10000, value=2000, step=100
)

# --- Generate population ---
N = 10_000_000

# Base: blend between uniform and normal
uniform_pop = np.random.uniform(-1, 1, N)
normal_pop = np.random.normal(0, 1, N)
base = (1 - uniformity) * uniform_pop + uniformity * normal_pop

# Add skew if requested
if shape > 0:
    skew_factor = shape * 2
    exp_component = np.random.exponential(1, N) - 1
    pop = (1 - shape) * base + shape * exp_component
else:
    pop = base

pop_mean = np.mean(pop)

# --- Sampling Distribution ---
sample_means = [np.mean(np.random.choice(pop, n, replace=True)) for _ in range(num_samples)]
sample_mean = np.mean(sample_means)
sample_sd = np.std(sample_means)
pop_sd = np.std(pop)
se_theoretical = pop_sd / np.sqrt(n)


# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].hist(pop, bins=50, color="#f39c6b", alpha=0.7)
axes[0].axvline(pop_mean, color="black", linestyle="--", label=f"μ ≈ {pop_mean:.2f}")
axes[0].set_title("Population Distribution")
axes[0].legend()

count, bins, _ = axes[1].hist(sample_means, bins=30, color="blue", alpha=0.5, density = True)
axes[1].axvline(np.mean(sample_means), color="black", linestyle="--", label=f"E(ȳ) ≈ {np.mean(sample_means):.2f}")
# Theoretical normal curve
x = np.linspace(min(bins), max(bins), 200)
pdf = stats.norm.pdf(x, loc=pop_mean, scale=se_theoretical)
axes[1].plot(x, pdf, 'k-', lw=2, label="Theoretical Normal")
axes[1].set_title("Sampling Distribution of the Mean")
axes[1].legend()

# 3. Q–Q Plot (normality check)
stats.probplot(sample_means, dist="norm", plot=axes[2])
axes[2].get_lines()[1].set_color("black")
axes[2].set_title("Q–Q Plot vs Normal")

plt.tight_layout()
st.pyplot(fig)


# Normality metrics
sk = stats.skew(sample_means)
ku = stats.kurtosis(sample_means)


st.markdown("### Distribution Metrics")

# Arrange in two rows
colA, colB, colC, colD = st.columns(4)
colA.metric("Population Mean (μ)", f"{pop_mean:.4f}")
colB.metric("Simulated Mean (ȳ̄)", f"{sample_mean:.4f}")
colC.metric("Theoretical SE (σ/√n)", f"{se_theoretical:.4f}")

colD.metric("Simulated SD (s₍ȳ₎)", f"{sample_sd:.4f}")

st.markdown("### Normality Metrics")
col1, col2 = st.columns(2)
col1.metric("Skewness", f"{stats.skew(sample_means):.3f}")
col2.metric("Kurtosis", f"{stats.kurtosis(sample_means):.3f}")


# --- Discussion ---
st.markdown(
    f"""
### Observations
- The **mean of the sampling distribution** (≈ {np.mean(sample_means):.2f}) stays close to the **population mean** (≈ {pop_mean:.2f}).
- As **n increases**, the sampling distribution becomes **narrower** (smaller variance).
- As the shape slider moves right, the population becomes **more skewed**, yet the **sampling distribution becomes more symmetric** for large n — illustrating the **Central Limit Theorem**.
"""
)
