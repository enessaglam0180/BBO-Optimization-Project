import numpy as np
import pandas as pd
from BBO import BBO
from PSO import PSO
from GWO import GWO
from benchmark_functions import solar_pv_cost

# --- AYARLAR ---
NUM_RUNS = 15       # Her algoritma kaç kez çalışsın? (İdeal: 30, Test için: 10-15)
POP_SIZE = 50       # Popülasyon
MAX_ITER = 200      # İterasyon

# Solar PV Sınırları (Daraltılmış)
bounds = [
    (0.7, 0.8), (1e-7, 1e-6), (0.01, 0.05), (40, 60), (1.4, 1.6)
]

algorithms = {
    "BBO": BBO,
    "PSO": PSO,
    "GWO": GWO
}

results_table = []

print(f"📊 İstatistik Toplama Başladı ({NUM_RUNS} tur)...")

for name, AlgoClass in algorithms.items():
    print(f"\n--- {name} Koşuluyor ---")
    fitness_values = []
    
    for i in range(NUM_RUNS):
        # İlerleme çubuğu gibi çıktı verelim
        print(f"\rRun {i+1}/{NUM_RUNS}...", end="")
        
        optimizer = AlgoClass(solar_pv_cost, bounds, POP_SIZE, MAX_ITER)
        _, best_fit, _ = optimizer.optimize()
        fitness_values.append(best_fit)
    
    # İstatistikleri Hesapla
    best_val = np.min(fitness_values)
    worst_val = np.max(fitness_values)
    mean_val = np.mean(fitness_values)
    std_val = np.std(fitness_values)
    
    # Tabloya Ekle
    results_table.append({
        "Algorithm": name,
        "Best (En İyi)": f"{best_val:.6f}",
        "Worst (En Kötü)": f"{worst_val:.6f}",
        "Mean (Ortalama)": f"{mean_val:.6f}",
        "Std Dev (Sapma)": f"{std_val:.2e}" 
    })
    print(f"\nTamamlandı. Ortalama Hata: {mean_val:.6f}")

# --- KAYDETME ---
df = pd.DataFrame(results_table)
df.to_csv("final_results.csv", index=False)
print("\n✅ Tüm sonuçlar 'final_results.csv' dosyasına kaydedildi!")
print("Excel ile açıp Raporundaki 'Results' tablosuna yapıştırabilirsin.")