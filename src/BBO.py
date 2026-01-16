import numpy as np
import copy

class BBO:
    """
    Beaver Behavior Optimizer (BBO) - 2025
    Based on the paper: "Beaver behavior optimizer: A novel metaheuristic algorithm..."
    """
    
    def __init__(self, objective_func, bounds, pop_size=30, max_iter=500):
        """
        Başlangıç parametrelerini ayarlar.
        :param objective_func: Optimize edilecek maliyet fonksiyonu (Cost Function)
        :param bounds: Değişkenlerin alt ve üst sınırları (lb, ub) -> [(min, max), (min, max)...]
        :param pop_size: Kunduz popülasyon sayısı (Varsayılan: 30)
        :param max_iter: Maksimum iterasyon sayısı (Varsayılan: 500)
        """
        self.func = objective_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # Sınırları ayır (Lower Bound ve Upper Bound vektörleri)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        
        # Popülasyonu başlat (Eq. 1 & 2)
        # Initialization of beaver population with random materials
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        
        # En iyi çözümü saklamak için değişkenler
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = [] # Yakınsama grafiği için kayıt

    def optimize(self):
        """
        Algoritmanın ana döngüsünü çalıştırır.
        """
        # İlk fitness hesaplaması
        for i in range(self.pop_size):
            self.fitness[i] = self.func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()
        
        # Ana İterasyon Döngüsü
        for t in range(self.max_iter):
            
            # --- Dam-Phase Factor (Eq. 3) ---
            # Zamanla 0'dan 1'e artan geçiş faktörü. 
            # Exploration (Keşif) ve Exploitation (Sömürü) dengesini sağlar.
            D = np.sin((np.pi * (t + 1)) / (2 * self.max_iter))
            
            # Random kontrol sayısı
            r1 = np.random.rand()
            
            # Popülasyonu fitness değerine göre sırala (En iyi kunduzlar başa)
            sorted_indices = np.argsort(self.fitness)
            sorted_pop = self.population[sorted_indices]
            
            # Architects (Mimarlar) ve Prospectors (Arayıcılar) ayrımı
            # Makaleye göre en iyi %25 Mimar olur [cite: 169]
            num_architects = int(0.25 * self.pop_size)
            
            # Yeni pozisyonları tutacak geçici dizi
            new_population = np.zeros_like(self.population)
            
            # --- Phase Selection ---
            if r1 <= D:
                # ---------------------------------------------------------
                # EXPLOITATION PHASE (Dam Maintenance) [cite: 187-195]
                # ---------------------------------------------------------
                # Bu fazda tüm popülasyon "Architect" gibi davranır ve
                # en iyi çözüme (Lider Kunduz) yakınsamaya çalışır.
                
                for i in range(self.pop_size):
                    # Rastgele başka bir kunduz seç (k != i)
                    k = np.random.randint(0, self.pop_size)
                    while k == i:
                        k = np.random.randint(0, self.pop_size)
                    
                    r7 = np.random.rand()
                    r8 = np.random.rand()
                    
                    # Eq. (6) - Baraj onarımı ve iyileştirme denklemi
                    # X_new = X_current + r7*(X_neighbor - X_current) + r8*(X_best - X_current)
                    new_pos = self.population[i] + \
                              r7 * (self.population[k] - self.population[i]) + \
                              r8 * (self.best_solution - self.population[i])
                    
                    new_population[i] = new_pos

            else:
                # ---------------------------------------------------------
                # EXPLORATION PHASE (Material Gathering) [cite: 162-186]
                # ---------------------------------------------------------
                # Popülasyon Mimarlar ve Arayıcılar olarak ikiye bölünür.
                
                # 1. Architects Update (Eq. 4)
                for i in range(num_architects):
                    # Rastgele başka bir mimar seç
                    k = np.random.randint(0, num_architects)
                    
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    
                    # Eğer r2 < 0.5 ise komşudan öğren, yoksa yerinde kal
                    if r2 < 0.5:
                        new_pos = sorted_pop[i] + r3 * (sorted_pop[k] - sorted_pop[i])
                    else:
                        new_pos = sorted_pop[i]
                    
                    new_population[i] = new_pos # Not: Bu yeni konumlar sorted sırasına göre değil, orijinal indekse göre geri yazılmalı ama basitlik için direkt atıyoruz.
                
                # 2. Prospectors Update (Eq. 5)
                for i in range(num_architects, self.pop_size):
                    # Rastgele bir mimar seç (Öğrenmek için)
                    k = np.random.randint(0, num_architects)
                    
                    r4 = np.random.rand()
                    r5 = np.random.rand()
                    # Gaussian random number (Mean=0, Var=1)
                    r6 = np.random.randn() 
                    
                    # Levy Flight benzeri bir sıçrama terimi (Makaledeki cos terimi)
                    # perturbation = r6 * cos(pi*t / 2T) * (ub - lb) / 10
                    perturbation = r6 * np.cos((np.pi * (t + 1)) / (2 * self.max_iter)) * (self.ub - self.lb) / 10.0
                    
                    # Eq. (5)
                    if r4 < 0.5:
                        learning_term = r5 * (sorted_pop[k] - sorted_pop[i])
                    else:
                        learning_term = 0
                        
                    new_pos = sorted_pop[i] + learning_term + perturbation
                    new_population[i] = new_pos

            # --- Sınır Kontrolü ve Fitness Güncelleme ---
            for i in range(self.pop_size):
                # 1. Sınır kontrolü (Clamping)
                new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                
                # 2. Yeni konumun fitness değerini hesapla
                new_fitness = self.func(new_population[i])
                
                # 3. ÖNEMLİ DÜZELTME:
                # new_population[i], aslında sorted_pop[i]'nin (yani i. sıradaki en iyinin) çocuğudur.
                # Bu yüzden kıyaslamayı orijinal indisteki rastgele bireyle değil,
                # sıralamadaki o "ebeveyn" (parent) ile yapmalıyız.
                
                original_index = sorted_indices[i] # Bu bireyin orijinal listedeki yeri
                
                # Eğer çocuk ebeveynden iyiyse, orijinal listedeki yerini güncelle
                if new_fitness < self.fitness[original_index]:
                    self.fitness[original_index] = new_fitness
                    self.population[original_index] = new_population[i]
                    
                    # Global en iyiyi kontrol et
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_population[i].copy()
            
            # Kayıt tut
            self.convergence_curve.append(self.best_fitness)
            
            # Her 50 adımda bir bilgi ver
            if (t+1) % 50 == 0:
                print(f"Iter: {t+1}, Best Fitness: {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness, self.convergence_curve