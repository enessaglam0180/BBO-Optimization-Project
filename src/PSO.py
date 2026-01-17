import numpy as np

class PSO:
    def __init__(self, objective_func, bounds, pop_size=30, max_iter=500):
        self.func = objective_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        
        # Parçacıkları ve Hızları Başlat
        self.X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.V = np.zeros_like(self.X)
        
        # En iyi konumlar (Personal Best)
        self.P_best = self.X.copy()
        self.fitness = np.full(self.pop_size, float('inf'))
        self.P_best_fit = np.full(self.pop_size, float('inf'))
        
        # Global En İyi (Global Best)
        self.g_best = None
        self.g_best_fit = float('inf')
        self.convergence_curve = []

    def optimize(self):
        # Parametreler (Standart PSO ayarları)
        w = 0.7  # Atalet ağırlığı (Inertia weight)
        c1 = 1.5 # Bilişsel katsayı
        c2 = 1.5 # Sosyal katsayı
        
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Sınır kontrolü
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                
                # Fitness hesapla
                fit = self.func(self.X[i])
                
                # Personal Best Güncelleme
                if fit < self.P_best_fit[i]:
                    self.P_best_fit[i] = fit
                    self.P_best[i] = self.X[i].copy()
                    
                    # Global Best Güncelleme
                    if fit < self.g_best_fit:
                        self.g_best_fit = fit
                        self.g_best = self.X[i].copy()
            
            # Hız ve Pozisyon Güncelleme
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # PSO Hız Denklemi
                self.V[i] = w * self.V[i] + \
                            c1 * r1 * (self.P_best[i] - self.X[i]) + \
                            c2 * r2 * (self.g_best - self.X[i])
                
                # Pozisyon Güncelleme
                self.X[i] = self.X[i] + self.V[i]
            
            self.convergence_curve.append(self.g_best_fit)
            
        return self.g_best, self.g_best_fit, self.convergence_curve