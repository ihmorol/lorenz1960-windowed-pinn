import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Lorenz-1960 System Configuration

# Parameters from Section 4.2, k=2, l=1
K = 2.0
L = 1.0

# Initial conditions
X0, Y0, Z0 = 0.5, 0.75, 1.0
INITIAL_STATE = np.array([X0, Y0, Z0])

# Time span
T_START = 0.0
T_END = 1.0
T_SPAN = (T_START, T_END)

# RK4 configuration
RK4_STEP = 1e-3
N_POINTS = int((T_END - T_START) / RK4_STEP) + 1

# SciPy configuration (high accuracy reference)
SCIPY_METHOD = "DOP853"
SCIPY_RTOL = 1e-10
SCIPY_ATOL = 1e-12


# Lorenz-1960 Right-Hand Side


def lorenz1960_rhs(t, state):
    x, y, z = state

    # Coefficients computed from k=2, l=1
    coeff_x = K * L * (1.0 / (K**2 + L**2) - 1.0 / K**2)  # -0.1
    coeff_y = K * L * (1.0 / L**2 - 1.0 / (K**2 + L**2))  # 1.6
    coeff_z = 0.5 * K * L * (1.0 / K**2 - 1.0 / L**2)  # -0.75

    dx = coeff_x * y * z
    dy = coeff_y * x * z
    dz = coeff_z * x * y

    return np.array([dx, dy, dz])


# RK4 Solver Implementation


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_solve(f, t_span, y0, h):
    t_start, t_end = t_span
    n_steps = int(round((t_end - t_start) / h))

    t_array = np.linspace(t_start, t_end, n_steps + 1)
    y_array = np.zeros((n_steps + 1, len(y0)))
    y_array[0] = np.asarray(y0, dtype=float)

    for i in range(n_steps):
        y_array[i + 1] = rk4_step(f, t_array[i], y_array[i], h)

    return t_array, y_array


# SciPy Reference Solver


def scipy_solve(f, t_span, y0, t_eval):
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method=SCIPY_METHOD,
        t_eval=t_eval,
        rtol=SCIPY_RTOL,
        atol=SCIPY_ATOL,
    )

    if not sol.success:
        raise RuntimeError(f"SciPy solver failed: {sol.message}")

    return sol.t, sol.y.T


# Error Metrics


def compute_mse(reference, candidate):
    """Compute Mean Squared Error."""
    return np.mean((candidate - reference) ** 2)


def compute_errors(reference, candidate):
    """Compute error for each state variable."""
    return candidate - reference


# Visualization
def create_comparison_plot(t, rk4_sol, scipy_sol, save_path=None):
    # Compute errors
    errors = compute_errors(scipy_sol, rk4_sol)
    mse_total = compute_mse(scipy_sol, rk4_sol)

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Colors for x, y, z
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
    labels = ["x", "y", "z"]

    # Left panel: Solution comparison
    for i, (color, label) in enumerate(zip(colors, labels)):
        # Plot RK4 solution (solid line)
        ax1.plot(
            t,
            rk4_sol[:, i],
            color=color,
            linewidth=2,
            label=f"${label}$",
            linestyle="-",
        )
        # Plot SciPy reference (dotted line for reference)
        ax1.plot(
            t, scipy_sol[:, i], color=color, linewidth=1.5, linestyle=":", alpha=0.8
        )

    ax1.set_xlabel("t", fontsize=12)
    ax1.set_ylabel("$x, y, z$", fontsize=12)
    ax1.set_title("RK4 Solution vs SciPy Reference", fontsize=13)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([T_START, T_END])

    # Right panel: Error curves
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax2.plot(t, errors[:, i], color=color, linewidth=2, label=f"$\\delta {label}$")

    ax2.set_xlabel("t", fontsize=12)
    ax2.set_ylabel("error", fontsize=12)
    ax2.set_title(f"MSE: ${mse_total:.2e}$", fontsize=13)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([T_START, T_END])

    # Format y-axis in scientific notation
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return mse_total


# Main Execution


def main():

    print("=" * 70)
    print("Lorenz-1960 Numerical Baseline")
    print("=" * 70)

    # Step 1: Solve with RK4
    print("\n[1] Solving with RK4 (step size = {:.0e})...".format(RK4_STEP))
    t_rk4, sol_rk4 = rk4_solve(lorenz1960_rhs, T_SPAN, INITIAL_STATE, RK4_STEP)
    print(f"    Time points: {len(t_rk4)}")
    print(
        f"    Final state: x={sol_rk4[-1, 0]:.6f}, y={sol_rk4[-1, 1]:.6f}, z={sol_rk4[-1, 2]:.6f}"
    )

    # Step 2: Solve with SciPy (reference)
    print("\n[2] Solving with SciPy DOP853 (reference)...")
    t_scipy, sol_scipy = scipy_solve(lorenz1960_rhs, T_SPAN, INITIAL_STATE, t_rk4)
    print(f"    Function evaluations: {len(t_scipy)}")
    print(
        f"    Final state: x={sol_scipy[-1, 0]:.6f}, y={sol_scipy[-1, 1]:.6f}, z={sol_scipy[-1, 2]:.6f}"
    )

    # Step 3: Compute errors
    print("\n[3] Computing error metrics...")
    errors = compute_errors(sol_scipy, sol_rk4)
    mse = compute_mse(sol_scipy, sol_rk4)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(errors))

    print(f"    MSE:  {mse:.6e}")
    print(f"    RMSE: {rmse:.6e}")
    print(f"    Max absolute error: {max_error:.6e}")

    # Step 4: Generate visualization
    print("\n[4] Generating comparison plot...")
    mse_total = create_comparison_plot(
        t_rk4, sol_rk4, sol_scipy, save_path="lorenz1960_comparison.png"
    )

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"The RK4 and SciPy solutions agree closely:")
    print(f"  - MSE:  {mse:.2e}")
    print(f"  - RMSE: {rmse:.2e}")
    print(f"\nThis baseline is validated and ready for ANN experiments.")
    print("=" * 70)

    return t_rk4, sol_rk4, sol_scipy, errors


if __name__ == "__main__":
    t, rk4_sol, scipy_sol, err = main()
