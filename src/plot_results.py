import numpy as np
import matplotlib.pyplot as plt
from BBO import BBO
from benchmark_functions import solar_pv_cost, V_exp, I_exp

# --- AYARLAR ---
# Tablodaki sonucuna (0.0129) yakın bir sonuç bulmak için algoritmayı çalıştıracağız.
bounds = [(0.7, 0.8), (1e-7, 1e-6), (0.01, 0.05), (40, 60), (1.4, 1.6)]
optimizer = BBO(solar_pv_cost, bounds, pop_size=50, max_iter=200) # Hızlı olsun diye 200 iterasyon

print("🔄 Tabloyla uyumlu grafik üretiliyor...")
best_sol, best_rmse, _ = optimizer.optimize()

# --- PARAMETRELERİ AYIKLA ---
I_ph, I_sd, R_s, R_sh, n = best_sol

# --- TAHMİN EĞRİSİNİ HESAPLA ---
k = 1.3806503e-23
q = 1.60217646e-19
T = 33 + 273.15
VT = (n * k * T) / q

I_pred = []
for V in V_exp:
    I_est = I_ph
    for _ in range(10): # Newton-Raphson
        try:
            exp_val = np.exp((V + I_est * R_s) / VT)
        except:
            exp_val = np.exp(100)
        
        f_val = I_ph - I_sd * (exp_val - 1) - (V + I_est * R_s) / R_sh - I_est
        df_val = -I_sd * (R_s / VT) * exp_val - (R_s / R_sh) - 1
        
        if df_val == 0: break
        I_next = I_est - f_val / df_val
        I_est = I_next
    I_pred.append(I_est)

# RMSE'yi tekrar hesapla (Kontrol amaçlı)
final_rmse = np.sqrt(np.mean((np.array(I_pred) - I_exp)**2))

# --- ÇİZİM ---
plt.figure(figsize=(10, 6))
plt.scatter(V_exp, I_exp, color='red', label='Experimental Data', zorder=5)
plt.plot(V_exp, I_pred, color='blue', linewidth=2, label=f'BBO Prediction (RMSE={final_rmse:.4f})')

plt.title('I-V Characteristic Curve (BBO Optimized)', fontsize=14)
plt.xlabel('Voltage (V)', fontsize=12)
plt.ylabel('Current (A)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("Solar_PV_Result_Consistent.png", dpi=300)
plt.show()

print(f"Bu grafikteki RMSE değeri ({final_rmse:.4f}")