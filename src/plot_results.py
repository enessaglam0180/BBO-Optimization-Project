import numpy as np
import matplotlib.pyplot as plt
from benchmark_functions import V_exp, I_exp

# Senin az önce bulduğun "En İyi" sonuçları buraya elle yazıyoruz.
# (Kodun her seferinde çalışıp saatlerce beklemesine gerek yok, sonucu aldık zaten)
best_params = [0.761230, 0.0000001896, 0.036394, 14.870099, 1.502699]

I_ph = best_params[0]
I_sd = best_params[1]
R_s  = best_params[2]
R_sh = best_params[3]
n    = best_params[4]

# Fiziksel Sabitler
k = 1.3806503e-23
q = 1.60217646e-19
T = 33 + 273.15
VT = (n * k * T) / q

# Tahmin Eğrisini Hesapla (Senin Algoritman)
I_pred = []
for V in V_exp:
    I_est = I_ph
    for _ in range(10):
        try:
            exp_val = np.exp((V + I_est * R_s) / VT)
        except:
            exp_val = np.exp(100) # Overflow koruması
            
        f_val = I_ph - I_sd * (exp_val - 1) - (V + I_est * R_s) / R_sh - I_est
        df_val = -I_sd * (R_s / VT) * exp_val - (R_s / R_sh) - 1
        
        if df_val == 0: break
        I_next = I_est - f_val / df_val
        I_est = I_next
    I_pred.append(I_est)

# --- GRAFİK ÇİZİMİ ---
plt.figure(figsize=(10, 6))

# 1. Gerçek Veriler (Kırmızı Noktalar)
plt.scatter(V_exp, I_exp, color='red', label='Gerçek Veri (R.T.C France)', zorder=5)

# 2. Senin BBO Tahminin (Mavi Çizgi)
plt.plot(V_exp, I_pred, color='blue', linewidth=2, label='BBO Tahmini (RMSE=0.011)')

plt.title('Solar PV Parametre Tahmini: BBO Algoritması', fontsize=14)
plt.xlabel('Voltaj (V)', fontsize=12)
plt.ylabel('Akım (A)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# Grafiği kaydet
plt.savefig("Solar_PV_Result.png", dpi=300)
plt.show()

print("Grafik 'Solar_PV_Result.png' olarak kaydedildi.")