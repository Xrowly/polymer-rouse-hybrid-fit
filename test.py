import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# ΔΕΔΟΜΕΝΑ
# =============================================================================
# Load your CSV
df = pd.read_csv("data.csv", encoding="utf-8-sig")

omega = df["ω (s^-1)"].tolist()
G_prime = df["G' (Pa)"].tolist()
G_double_prime = df["G'' (Pa)"].tolist()

M = len(omega)
Np = 6  # number of modes
a = 2   # exponent in λ_p calculation
plist = np.ones(Np)

# =============================================================================
# ΣΥΝΑΡΤΗΣΕΙΣ
# =============================================================================
def find_l(max_p, lamda1):
    return [lamda1 / (p**a) for p in range(1, max_p+1)]

def eta(max_p, eta1, l_list):
    return [eta1 * l_list[0] / sum(l_list[:p]) for p in range(1, max_p+1)]

def G_Theta_per_mode(N, max_p, eta_list, l_list):
    G_prime_modes = [[0]*max_p for _ in range(N)]
    G_double_prime_modes = [[0]*max_p for _ in range(N)]
    for i in range(max_p):
        for p in range(N):
            denom = 1 + (l_list[p] * omega[i])**2
            G_prime_modes[p][i] = eta_list[p] * l_list[p] * omega[i]**2 / denom
            G_double_prime_modes[p][i] = eta_list[p] * omega[i] / denom
    return G_prime_modes, G_double_prime_modes

def cost(G,M):
    G1,G2=G
    kappa=[]
    for i in range(0,M):
        err1=(1-(G1[i]/G_prime[i]))*(1-(G1[i]/G_prime[i]))
        err2=(1-(G2[i]/G_double_prime[i]))*(1-(G2[i]/G_double_prime[i]))
        kappa.append(err1+err2)
    return sum(kappa)

# =============================================================================
# AUTOMATIC η₁ REFINEMENT
# =============================================================================
def auto_refine_eta(l1, initial_eta_values, target_cost=0.001, max_steps=20):
    current_eta_list = sorted(initial_eta_values)
    best_global = None
    history = []
    all_eta = []
    all_K = []

    for step_num in range(max_steps):
        step_results = []
        l_list = find_l(Np, l1)

        for eta_val in current_eta_list:
            eta_list = eta(Np, eta_val, l_list)
            Gp_modes, Gpp_modes = G_Theta_per_mode(Np, M, eta_list, l_list)
            Gp_total = [sum(col) for col in zip(*Gp_modes)]
            Gpp_total = [sum(col) for col in zip(*Gpp_modes)]
            current_K = cost([Gp_total, Gpp_total], M)

            step_results.append({
                'η₁': eta_val, 'Κ': current_K,
                "G'_modes": Gp_modes,
                "G''_modes": Gpp_modes,
                "G'_th": Gp_total,
                "G''_th": Gpp_total,
                'λ_p': l_list,
                'η_p': eta_list
            })

            all_eta.append(eta_val)
            all_K.append(current_K)

        res = min(step_results, key=lambda x: x['Κ'])
        history.append(res)

        if best_global is None or res['Κ'] < best_global['Κ']:
            best_global = res

        if best_global['Κ'] <= target_cost:
            break

        best_eta = res['η₁']
        lower = max(best_eta - best_eta / 2, 1e-12)
        upper = best_eta + best_eta / 2
        current_eta_list = [round(x,6) for x in np.linspace(lower, upper, 6)]

    return best_global, history, all_eta, all_K

# =============================================================================
# AUTOMATIC Λ₁ REFINEMENT
# =============================================================================
def auto_refine_l1(initial_l1_values, eta1_best, target_cost=0.001, max_steps=20):
    current_l1_list = sorted(initial_l1_values)
    best_global = None
    history = []

    for step_num in range(max_steps):
        step_results = []

        for l1_val in current_l1_list:
            l_list = find_l(Np, l1_val)
            eta_list = eta(Np, eta1_best, l_list)
            Gp_modes, Gpp_modes = G_Theta_per_mode(Np, M, eta_list, l_list)
            Gp_total = [sum(col) for col in zip(*Gp_modes)]
            Gpp_total = [sum(col) for col in zip(*Gpp_modes)]
            current_K = cost([Gp_total, Gpp_total], M)

            step_results.append({
                'Λ₁': l1_val, 'Κ': current_K,
                "G'_modes": Gp_modes,
                "G''_modes": Gpp_modes,
                "G'_th": Gp_total,
                "G''_th": Gpp_total,
                'λ_p': l_list,
                'η_p': eta_list
            })

        res = min(step_results, key=lambda x: x['Κ'])
        history.append(res)

        if best_global is None or res['Κ'] < best_global['Κ']:
            best_global = res

        if best_global['Κ'] <= target_cost:
            break

        best_l1 = res['Λ₁']
        lower = max(best_l1 - best_l1 / 2, 1e-12)
        upper = best_l1 + best_l1 / 2
        current_l1_list = [round(x,6) for x in np.linspace(lower, upper, 6)]

    return best_global, history

# =============================================================================
# MUTUAL Λ₁-η₁ OPTIMIZATION
# =============================================================================
def mutual_optimize_l1_eta(
    l1_values_guess, eta_values_guess,
    Np, M, tol=1e-6, max_outer_steps=10
):
    l1_best = np.mean(l1_values_guess)
    eta_best = np.mean(eta_values_guess)
    best_global = None
    history_global = []

    all_eta = []
    all_K = []
    all_l1 = []       # store Λ₁ guesses per step
    all_K_l1 = []     # store corresponding Κ per Λ₁

    for outer_step in range(max_outer_steps):
        # Step 1: Optimize η₁ given current Λ₁
        best_eta_result, _, step_all_eta, step_all_K = auto_refine_eta(
            l1=l1_best,
            initial_eta_values=eta_values_guess
        )
        eta_best = best_eta_result['η₁']

        # Store η₁ → Κ
        all_eta.extend(step_all_eta)
        all_K.extend(step_all_K)

        # Step 2: Optimize Λ₁ given new η₁
        best_l1_result, _ = auto_refine_l1(
            initial_l1_values=l1_values_guess,
            eta1_best=eta_best
        )
        l1_best = best_l1_result['Λ₁']

        # Store Λ₁ → Κ for all tested l1 values in this refinement
        step_l1_values = [res['Λ₁'] for res in _] if _ else [l1_best]  # fallback to best if _ empty
        step_K_values = [res['Κ'] for res in _] if _ else [best_l1_result['Κ']]
        all_l1.extend(step_l1_values)
        all_K_l1.extend(step_K_values)

        # Track best_global
        if best_global is None:
            best_global = best_l1_result
            improvement = np.inf
        else:
            improvement = abs(best_global['Κ'] - best_l1_result['Κ'])
            if improvement < tol:
                best_global = best_l1_result
                history_global.append(best_global)
                break
            best_global = best_l1_result

        history_global.append(best_global)
        print(f"Step {outer_step+1}: Λ₁={l1_best:.6f}, η₁={eta_best:.6f}, Κ={best_global['Κ']:.6f}")

    return best_global, history_global, eta_best, l1_best, all_eta, all_K, all_l1, all_K_l1



# =============================================================================
# RUN MUTUAL OPTIMIZATION
# =============================================================================
best, history, final_eta, final_l1, all_eta, all_K, all_l1, all_K_l1 = mutual_optimize_l1_eta(
    l1_values_guess=[0.1, 1, 10, 20, 50, 100],
    eta_values_guess=[1, 10, 100, 1000, 10000, 20000],
    Np=Np,
    M=M,
    tol=1e-6,
    max_outer_steps=10
)


print(f"FINAL BEST: Λ₁ = {final_l1:.6f}, η₁ = {final_eta:.6f}, Κ = {best['Κ']:.6f}")

# Load original data again (if not already in memory)
df = pd.read_csv("data.csv", encoding="utf-8-sig")

# Append the fitted G' and G'' from the best result
df['G\'_fit'] = best["G'_th"]
df['G\'\'_fit'] = best["G''_th"]

# Append the optimal Λ₁ and η₁ for reference
df['Λ1_opt'] = ''
df['η1_opt'] = ''
df.loc[0, 'Λ1_opt'] = final_l1
df.loc[0, 'η1_opt'] = final_eta


# Save the augmented CSV
#df.to_csv("data_with_fit.csv", index=False, encoding="utf-8-sig")
#print("Fitted data appended and saved to 'data_with_fit.csv'.")
# =============================================================================
# PLOTS: G' & G'' ΜΕ MODES
# =============================================================================
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
for p in range(Np):
    plt.loglog(omega, best['G\'_modes'][p], '--', label=f'p={p+1}', alpha=0.7)
plt.loglog(omega, G_prime, 's', color='red', label="G' exp", ms=8)
plt.loglog(omega, best['G\'_th'], 'k-', lw=2.5, label="G' th")
plt.xlabel('ω (s⁻¹)'); plt.ylabel("G' (Pa)"); plt.legend(); plt.grid(True, which='both', ls=':')
plt.subplot(1,2,2)
for p in range(Np):
    plt.loglog(omega, best['G\'\'_modes'][p], '--', label=f'p={p+1}', alpha=0.7)
plt.loglog(omega, G_double_prime, 'o', color='blue', label="G'' exp", ms=8)
plt.loglog(omega, best['G\'\'_th'], 'k-', lw=2.5, label="G'' th")
plt.xlabel('ω (s⁻¹)'); plt.ylabel("G'' (Pa)"); plt.legend(); plt.grid(True, which='both', ls=':')
plt.suptitle(f"Λ₁={best['Λ₁']:.3f} s, η₁={best['η_p'][0]:.3f} Pa·s, Κ={best['Κ']:.6f}")
plt.tight_layout()
plt.savefig("Plots_Modes.png", dpi=300)
plt.show()

# =============================================================================
# Κ-η DIAGRAM
# =============================================================================
etasublists = [all_eta[i:i+6] for i in range(0, len(all_eta), 6)]
kappasublists = [all_K[i:i+6] for i in range(0, len(all_K), 6)]

# Select sublists 61 to 80 (0-based index 60 to 79)
selected_indices = range(0, 20)

headers = ['Sublist_index', 'η₁', 'Κ']
all_data = []

for idx in selected_indices:
    eta_sub = etasublists[idx]
    kappa_sub = kappasublists[idx]
    
    # Add all rows for this sublist
    for e, k in zip(eta_sub, kappa_sub):
        all_data.append({
            'Sublist_index': idx+1,  # 1-based index
            'η₁': e,
            'Κ': k
        })
    
    # Add a blank row
    all_data.append({col: '' for col in headers})
    
    # Add header row before next sublist
    all_data.append({col: col for col in headers})

# Convert to DataFrame
df_summary = pd.DataFrame(all_data)

# Save to CSV
df_summary.to_csv("K_vs_eta_summary_61_80_blocked.csv", index=False, encoding="utf-8-sig")
#print("CSV saved as 'K_vs_eta_summary_61_80_blocked.csv' with blank rows and headers between sublists.")


#plt.figure(figsize=(7,5))
#for i in range(0,len(etasublists)):
#    plt.plot(etasublists[i], kappasublists[i], 'o-', color='purple', markersize=6, linewidth=1.5)
#    plt.xlabel("η₁ (Pa·s)")
#    plt.ylabel("Κ")
#    plt.title(f"Κ-η₁ (Απλή Καμπύλη) {i}")
#    plt.grid(True, ls=':', alpha=0.6)
#    plt.savefig(f"K_eta_Modes{i}.png", dpi=300)
#    plt.show()


# =============================================================================
# FULL MODES DATA CSV
# =============================================================================

# Column headers for the data
headers = ['Λ₁','η₁','Mode','λ_p','η_p','ω',"G'_mode","G''_mode","G'_total","G''_total","Κ"]

# Initialize list to store all data
all_matrix_data = []

# Loop through mutual optimization history
for record in history:
    Λ1 = record.get('Λ₁', best['Λ₁'])
    η1 = record.get('η₁', best['η_p'][0])
    Gp_modes = record.get("G'_modes", best["G'_modes"])
    Gpp_modes = record.get("G''_modes", best["G''_modes"])
    λ_list = record.get('λ_p', best['λ_p'])
    η_list = record.get('η_p', best['η_p'])
    Gp_total = record.get("G'_th", best["G'_th"])
    Gpp_total = record.get("G''_th", best["G''_th"])
    Κ_val = record.get('Κ', best['Κ'])
    
    # 1. Insert a row showing Λ₁ value
    all_matrix_data.append({'Λ₁': f"Λ₁={Λ1}"})
    
    for p in range(Np):
        # 2. Insert a blank row before each mode
        all_matrix_data.append({col: '' for col in headers})
        
        # 3. Insert column headers
        all_matrix_data.append({col: col for col in headers})
        
        # 4. Insert the mode × frequency data
        for i in range(M):
            all_matrix_data.append({
                'Λ₁': Λ1,
                'η₁': η1,
                'Mode': p+1,
                'λ_p': λ_list[p],
                'η_p': η_list[p],
                'ω': omega[i],
                "G'_mode": Gp_modes[p][i],
                "G''_mode": Gpp_modes[p][i],
                "G'_total": Gp_total[i],
                "G''_total": Gpp_total[i],
                "Κ": Κ_val
            })

# Convert to DataFrame
df_matrix = pd.DataFrame(all_matrix_data)

# Save to CSV
#df_matrix.to_csv("Full_Matrix_Fitting_blocked_modes.csv", index=False)
#print("Full matrix with blank rows and headers before each mode saved to 'Full_Matrix_Fitting_blocked_modes.csv'.")


# =============================================================================
# Κ-Λ₁ DIAGRAM (analogous to Κ-η₁)
# =============================================================================
# Split into sublists of Np values per sweep (like we did for eta)
l1_sublists = [all_l1[i:i+6] for i in range(0, len(all_l1), 6)]
kappa_l1_sublists = [all_K_l1[i:i+6] for i in range(0, len(all_K_l1), 6)]

# Select range of sublists if desired
selected_indices_l1 = range(0, len(l1_sublists))  # adjust as needed

headers_l1 = ['Sublist_index', 'Λ₁', 'Κ']
all_data_l1 = []

for idx in selected_indices_l1:
    l1_sub = l1_sublists[idx]
    kappa_sub = kappa_l1_sublists[idx]

    for l_val, k_val in zip(l1_sub, kappa_sub):
        all_data_l1.append({
            'Sublist_index': idx+1,  # 1-based
            'Λ₁': l_val,
            'Κ': k_val
        })

    # blank row and header row
    all_data_l1.append({col: '' for col in headers_l1})
    all_data_l1.append({col: col for col in headers_l1})

df_l1_summary = pd.DataFrame(all_data_l1)

# Optional: save separately
df_l1_summary.to_csv("K_vs_l1_summary.csv", index=False, encoding="utf-8-sig")
import os

# Folder for Lambda plots
folder_name = "Lambda_Plots"
os.makedirs(folder_name, exist_ok=True)  # create folder if it doesn't exist

for i in range(len(l1_sublists)):
    # Get sublist
    l_sub = l1_sublists[i]
    k_sub = kappa_l1_sublists[i]

    # Sort Λ₁ and Κ together
    l_sorted, k_sorted = zip(*sorted(zip(l_sub, k_sub)))

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(l_sorted, k_sorted, 'o-', color='green', markersize=6, linewidth=1.5)
    plt.xlabel("Λ₁ (s)")
    plt.ylabel("Κ")
    plt.title(f"Κ vs Λ₁, Sublist {i+1}")
    plt.grid(True, ls=':', alpha=0.6)
    plt.tight_layout()

    # Save inside folder
    plt.savefig(os.path.join(folder_name, f"k_L{i+1}.png"), dpi=300)
    plt.close()

