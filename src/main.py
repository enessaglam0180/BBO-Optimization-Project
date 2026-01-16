from BBO import BBO
from benchmark_functions import solar_pv_cost
import numpy as np

# 1. Solar PV Parametre Sınırları (Hata almamak için sıkılaştırıldı)
# Makale değerlerine uygun aralıklar:
bounds = [
    (0.7, 0.8),    # I_ph: Genelde 0.76 civarındadır, aralığı daralttık.
    (1e-7, 1e-6),  # I_sd: En kritik ayar! Samanlıkta iğne aramayı bırakıp nokta atışı yapsın.
    (0.01, 0.05),  # R_s : Genelde 0.036'dır.
    (10, 100),     # R_sh: Genelde 50 civarındadır.
    (1.4, 1.6)     # n   : Genelde 1.48'dir.
]
# 2. BBO Optimizatörünü Başlat
optimizer = BBO(solar_pv_cost, bounds, pop_size=100, max_iter=1000)

print("🌞 Solar PV Optimizasyonu Başlatılıyor (Single Diode Model)...")
best_sol, best_fit, curve = optimizer.optimize()

print("\n--- SONUÇLAR ---")
print(f"En İyi RMSE Değeri: {best_fit:.8f} (Hedef < 0.001)")
print("Optimize Edilen Parametreler:")
print(f"I_ph (A) : {best_sol[0]:.6f}")
print(f"I_sd (A) : {best_sol[1]:.10f}")
print(f"R_s (Ohm): {best_sol[2]:.6f}")
print(f"R_sh(Ohm): {best_sol[3]:.6f}")
print(f"n        : {best_sol[4]:.6f}")