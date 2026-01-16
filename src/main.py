from BBO import BBO
from benchmark_functions import solar_pv_cost
import numpy as np

# 1. Solar PV Parametre Sınırları (Hata almamak için sıkılaştırıldı)
# Makale değerlerine uygun aralıklar:
bounds = [
    (0, 1),        # I_ph (Fotovoltaik Akım)
    (1e-12, 1e-5), # I_sd (Diyot Akımı - Çok hassas)
    (0.001, 1),    # R_s  (Seri Direnç - 0 olamaz)
    (10, 200),     # R_sh (Şönt Direnç - 0 olamaz, genelde 30-100 arasıdır)
    (1, 2)         # n    (İdealite Faktörü)
]

# 2. Algoritmayı Kur
# Popülasyonu 50, İterasyonu 1000 yaparsak sonuç makale kalitesinde olur.
optimizer = BBO(solar_pv_cost, bounds, pop_size=50, max_iter=1000)

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