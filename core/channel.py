import numpy as np

def apply_awgn(signal, snr_db):
    """Añade ruido blanco gaussiano"""
    # Potencia de la señal
    sig_power = np.mean(np.abs(signal)**2)
    # Potencia del ruido requerida
    noise_power = sig_power / (10**(snr_db/10))
    # Generar ruido complejo
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def apply_rayleigh(signal, snr_db, num_taps=1):
    """
    Simula canal multipath SISO (1 Tx -> 1 Rx).
    AHORA ES ALEATORIO PARA SER JUSTO CON MISO.
    """
    if num_taps == 1:
        h = np.array([1.0 + 0j]) 
    else:
        # --- CAMBIO IMPORTANTE: Generación Aleatoria (Rayleigh) ---
        # Antes usabas un array fijo [1.0, 0.1...], que es muy fácil.
        # Ahora generamos taps aleatorios complejos como en MISO.
        h = (np.random.randn(num_taps) + 1j*np.random.randn(num_taps))
        
        # Normalizar energía del canal a 1
        h = h / np.sqrt(np.sum(np.abs(h)**2))

    # Convolución
    signal_convolved = np.convolve(signal, h, mode='full')
    signal_convolved = signal_convolved[:len(signal)]

    # Añadir ruido AWGN
    signal_noisy = apply_awgn(signal_convolved, snr_db)
    
    return signal_noisy, h


def apply_miso_rayleigh(signal_ant1, signal_ant2, snr_db, num_taps=1):
    """
    Simula canal MISO (2 Tx -> 1 Rx) con dos caminos independientes.
    """
    if num_taps == 1:
        h1 = np.array([1.0 + 0j]) 
        h2 = np.array([1.0 + 0j])
    else:
        # Generar taps aleatorios para h1
        h1 = (np.random.randn(num_taps) + 1j*np.random.randn(num_taps))
        h1 = h1 / np.sqrt(np.sum(np.abs(h1)**2)) 
        
        # Generar taps aleatorios para h2 (INDEPENDIENTE)
        h2 = (np.random.randn(num_taps) + 1j*np.random.randn(num_taps))
        h2 = h2 / np.sqrt(np.sum(np.abs(h2)**2)) 

    # Convolución de cada antena
    rx1 = np.convolve(signal_ant1, h1, mode='full')
    rx2 = np.convolve(signal_ant2, h2, mode='full')
    
    # Superposición
    rx_combined = rx1 + rx2
    L = len(signal_ant1)
    rx_combined = rx_combined[:L]
    
    # Añadir Ruido
    rx_noisy = apply_awgn(rx_combined, snr_db)
    
    return rx_noisy, h1, h2