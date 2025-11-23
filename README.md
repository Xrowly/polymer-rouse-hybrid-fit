# Rouse Model Fitting for Viscoelastic Data (2025-2026)

Complete and robust fitting of the classical Rouse model (6 modes, λ_p ∝ p⁻²) to experimental storage modulus G' and loss modulus G'' vs angular frequency ω.

The two fitted parameters are:  
- **λ₁** → longest relaxation time (s)  
- **η₁** → viscosity contribution of the first mode (Pa·s)

A custom **hybrid mutual-refinement optimizer** (alternating zoom-in grid searches on λ₁ and η₁) is implemented. It consistently finds the **global minimum** and is faster and more reliable than standard SciPy global optimizers.

## Repository Contents

| File / Folder                  | What it is & What it does when you run it                                      |
|--------------------------------|---------------------------------------------------------------------------------|
| `data.csv`                     | Experimental data (ω, G', G'') – the 2025-2026 dataset                         |
| `test.py`                      | **Main script** – runs the hybrid optimizer, prints final parameters, creates: <br>• `Plots_Modes.png` (fit + individual modes) <br>• folder `Lambda_Plots/` with hundreds of diagnostic Κ-Λ₁ & Κ-η₁ curves <br>• several CSV files with full history and per-mode contributions |
| `minimizertest.py`             | **Benchmark script** – runs Nelder-Mead, Powell, L-BFGS-B, SLSQP, Differential Evolution and SHGO on the same data and prints a table proving that **none of them beat the hybrid method** (same K, but slower) |
| `organize_plots.sh`            | Bash script – groups the many diagnostic plots from `Lambda_Plots/` into folders of 20 (1-20, 21-40, …) for easy browsing |
| `Plots_Modes.png`              | Final result – experimental points + total fit + contribution of each of the 6 Rouse modes |
| `Lambda_Plots/`                | Auto-generated folder with diagnostic plots (Κ vs Λ₁ for every refinement step) |
| `K_vs_l1_summary.csv`          | Table with cost Κ vs tested Λ₁ values                                          |
| `requirements.txt`             | List of required Python packages                                               |

## Final Result (data.csv)
λ₁ = 4.24295 s
η₁ = 86.61   Pa·s
K  = 0.8497854   (relative squared error cost)

All serious global optimizers converge to the **exact same value within ±0.005 %** → this is the true global minimum.

## Requirements

```txt
numpy>=1.21
pandas>=1.3
scipy>=1.7
matplotlib>=3.5
```
### Install with:
```code
pip install numpy pandas scipy matplotlib
```
## How to Run – Step by Step

### 1. Main fitting (recommended – does everything)
```code
python test.py
```
→ Prints live progress, final λ₁ & η₁, and generates all plots + CSV files.
### 2. Compare with SciPy optimizers (optional)
```code
python minimizertest.py
```
→ Shows a nice table with times and final K values.
### 3. Organize the hundreds of diagnostic plots (optional) (bash users)
```
chmod +x organize_plots.sh  # only the first time
./organize_plots.sh
```
→ Creates folders 1-20, 21-40, … inside Lambda_Plots/

# Customization

1. Change number of Rouse modes → edit Np = 6 in test.py
2. Use your own data → replace data.csv (keep column names: ω (s^-1), G' (Pa), G'' (Pa))
3. Tighter convergence → lower tol= in the mutual_optimize_l1_eta call

#Model Equations (exactly as implemented)
```text
λ_p = λ₁ / p²
η_p = η₁ · λ₁ / Σ_{k=1}^{p} λ_k

G'(ω)  = Σ_p  η_p λ_p ω² / (1 + (ω λ_p)²)
G''(ω) = Σ_p  η_p ω     / (1 + (ω λ_p)²)

Cost function Κ = Σ [ (G'_exp − G'_model)²/G'_exp² + (G''_exp − G''_model)²/G''_exp² ]
```
License
MIT – use, modify and share freely.
