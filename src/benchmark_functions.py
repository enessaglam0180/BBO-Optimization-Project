import numpy as np
import warnings

# R.T.C. France Verileri (Sabit)
V_exp = np.array([-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132, 0.2545, 0.2924, 0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.4590, 0.4784, 0.4960, 0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736, 0.5833, 0.5900])
I_exp = np.array([0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7555, 0.7540, 0.7505, 0.7465, 0.7385, 0.7280, 0.7140, 0.6975, 0.6745, 0.6475, 0.6175, 0.5790, 0.5350, 0.4850, 0.4240, 0.3600, 0.2745, 0.1770, 0.0870])

def solar_pv_cost(x):
    # Negatif değerleri engelle (Fiziksel İmkansızlık)
    if np.any(x < 0):
        return 1e10

    I_ph, I_sd, R_s, R_sh, n = x
    
    # R_sh veya n çok küçükse patlamayı önle
    if R_sh < 1e-5 or n < 0.1:
        return 1e10

    k = 1.3806503e-23
    q = 1.60217646e-19
    T = 33 + 273.15
    VT = (n * k * T) / q

    I_calc = np.zeros_like(V_exp)
    
    # Uyarıları gizle (Overflow uyarısı almamak için)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        for i in range(len(V_exp)):
            V = V_exp[i]
            I_est = I_ph 
            
            success = False
            for _ in range(10): # İterasyonu 10'a çıkardık
                try:
                    # Üstel ifadenin patlamasını önle (Overflow Protection)
                    exp_arg = (V + I_est * R_s) / VT
                    if exp_arg > 100: # Eğer sayı çok büyürse
                        exp_val = np.exp(100) * 1e5 # Yapay bir sınır koy
                    else:
                        exp_val = np.exp(exp_arg)

                    f_val = I_ph - I_sd * (exp_val - 1) - (V + I_est * R_s) / R_sh - I_est
                    df_val = -I_sd * (R_s / VT) * exp_val - (R_s / R_sh) - 1
                    
                    if abs(df_val) < 1e-10: # Türev sıfırsa dur
                        break
                        
                    I_next = I_est - f_val / df_val
                    
                    if np.isnan(I_next) or np.isinf(I_next):
                        break
                        
                    if abs(I_next - I_est) < 1e-6:
                        I_est = I_next
                        success = True
                        break
                    
                    I_est = I_next
                except:
                    break
            
            # Eğer Newton yöntemi başarısız olursa, o çözüm kötüdür
            if not success and (np.isnan(I_est) or np.isinf(I_est)):
                return 1e10 # Cezalandır
                
            I_calc[i] = I_est

    diff = I_calc - I_exp
    rmse = np.sqrt(np.mean(diff**2))
    
    if np.isnan(rmse):
        return 1e10
        
    return rmse