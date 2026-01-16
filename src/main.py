from BBO import BBO
import numpy as np

# 1. Test Fonksiyonu Tanımla (Örn: Sphere Function)
# Hedef: Tüm x'leri 0 yapıp sonucu 0 bulmak.
def sphere_function(x):
    return np.sum(x**2)

# 2. Parametreleri Ayarla
bounds = [(-100, 100) for _ in range(5)] # 5 boyutlu, -100 ile 100 arası
pop_size = 30
max_iter = 100

# 3. BBO Algoritmasını Başlat
optimizer = BBO(sphere_function, bounds, pop_size, max_iter)

# 4. Çalıştır
print("BBO Başlatılıyor...")
best_sol, best_fit, curve = optimizer.optimize()

print(f"En İyi Çözüm Konumu: {best_sol}")
print(f"En İyi Fitness Değeri: {best_fit}")