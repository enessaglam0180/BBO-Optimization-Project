import numpy as np

class GWO:
    def __init__(self, objective_func, bounds, pop_size=30, max_iter=500):
        self.func = objective_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        
        self.X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # İlk 3 lider (Alpha, Beta, Delta)
        self.Alpha_pos = np.zeros(self.dim)
        self.Alpha_score = float("inf")
        
        self.Beta_pos = np.zeros(self.dim)
        self.Beta_score = float("inf")
        
        self.Delta_pos = np.zeros(self.dim)
        self.Delta_score = float("inf")
        
        self.convergence_curve = []

    def optimize(self):
        for t in range(self.max_iter):
            
            # Liderleri Belirle
            for i in range(self.pop_size):
                # Sınır kontrolü
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                
                fitness = self.func(self.X[i])
                
                if fitness < self.Alpha_score:
                    self.Alpha_score = fitness
                    self.Alpha_pos = self.X[i].copy()
                elif fitness < self.Beta_score:
                    self.Beta_score = fitness
                    self.Beta_pos = self.X[i].copy()
                elif fitness < self.Delta_score:
                    self.Delta_score = fitness
                    self.Delta_pos = self.X[i].copy()
            
            # a parametresi 2'den 0'a lineer azalır
            a = 2 - t * (2 / self.max_iter)
            
            # Pozisyon Güncelleme (Eq 3.1 - 3.7 in GWO paper)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    
                    # Alpha'ya göre
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.Alpha_pos[j] - self.X[i, j])
                    X1 = self.Alpha_pos[j] - A1 * D_alpha
                    
                    # Beta'ya göre
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.Beta_pos[j] - self.X[i, j])
                    X2 = self.Beta_pos[j] - A2 * D_beta
                    
                    # Delta'ya göre
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.Delta_pos[j] - self.X[i, j])
                    X3 = self.Delta_pos[j] - A3 * D_delta
                    
                    # Ortalama
                    self.X[i, j] = (X1 + X2 + X3) / 3
            
            self.convergence_curve.append(self.Alpha_score)
            
        return self.Alpha_pos, self.Alpha_score, self.convergence_curve