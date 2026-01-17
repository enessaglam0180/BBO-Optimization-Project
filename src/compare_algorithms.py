import numpy as np
import matplotlib.pyplot as plt
from BBO import BBO
from PSO import PSO
from GWO import GWO
from benchmark_functions import solar_pv_cost

# --- AYARLAR ---
pop_size = 50
max_iter = 300  # Kıyaslama için 300 yeterli, hızlı olsun

# Parametre Sınırları (Az önce düzelttiğimiz sınırlar)
bounds = [
    (0.7, 0.8),    # I_ph
    (1e-7, 1e-6),  # I_sd
    (0.01, 0.05),  # R_s
    (40, 60),      # R_sh
    (1.4, 1.6)     # n
]

print("🚀 ALGORİTMALAR YARIŞIYOR...")

# 1. BBO Çalıştır
print("1. BBO Çalışıyor...")
bbo = BBO(solar_pv_cost, bounds, pop_size, max_iter)
_, bbo_best, bbo_curve = bbo.optimize()
print(f"   -> BBO Sonuç: {bbo_best:.6f}")

# 2. PSO Çalıştır
print("2. PSO Çalışıyor...")
pso = PSO(solar_pv_cost, bounds, pop_size, max_iter)
_, pso_best, pso_curve = pso.optimize()
print(f"   -> PSO Sonuç: {pso_best:.6f}")

# 3. GWO Çalıştır
print("3. GWO Çalışıyor...")
gwo = GWO(solar_pv_cost, bounds, pop_size, max_iter)
_, gwo_best, gwo_curve = gwo.optimize()
print(f"   -> GWO Sonuç: {gwo_best:.6f}")

# --- GRAFİK ÇİZİMİ (Convergence Curve) ---
plt.figure(figsize=(10, 6))

plt.plot(bbo_curve, label=f'BBO (Best: {bbo_best:.5f})', linewidth=2, color='red')
plt.plot(pso_curve, label=f'PSO (Best: {pso_best:.5f})', linewidth=1.5, linestyle='--', color='blue')
plt.plot(gwo_curve, label=f'GWO (Best: {gwo_best:.5f})', linewidth=1.5, linestyle='-.', color='green')

plt.title('Yakınsama Analizi: Solar PV Parametre Tahmini', fontsize=14)
plt.xlabel('İterasyon', fontsize=12)
plt.ylabel('Maliyet (RMSE)', fontsize=12)
plt.yscale('log') # Logaritmik ölçek farkları daha iyi gösterir
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.savefig("Comparison_Result.png", dpi=300)
plt.show()

print("\n✅ Karşılaştırma grafiği 'Comparison_Result.png' olarak kaydedildi.")