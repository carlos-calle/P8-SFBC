import numpy as np

def modulate_ofdm(symbols, n_fft, nc):
    """
    Empaqueta símbolos en subportadoras y aplica IFFT.
    symbols: Array de símbolos complejos (QPSK/QAM)
    n_fft: Tamaño total de la FFT
    nc: Subportadoras activas
    """
    num_symbols = len(symbols)
    # Número de bloques OFDM necesarios
    num_blocks = int(np.ceil(num_symbols / nc))
    
    # Rellenar con ceros si el último bloque no está lleno
    pad_len = num_blocks * nc - num_symbols
    if pad_len > 0:
        symbols = np.concatenate((symbols, np.zeros(pad_len)))
        
    ofdm_time_signal = []
    
    for i in range(num_blocks):
        # Extraer bloque de símbolos
        block = symbols[i*nc : (i+1)*nc]
        
        # Mapeo a entradas de IFFT
        ifft_input = np.zeros(n_fft, dtype=complex)
        
        # Mapeo simple: Frecuencias 0 a Nc-1 (Parte positiva)
        ifft_input[1:nc+1] = block # Dejamos DC en 0 vacío
        
        # IFFT
        time_sym = np.fft.ifft(ifft_input) * np.sqrt(n_fft) # Normalización de energía
        ofdm_time_signal.extend(time_sym)
        
    return np.array(ofdm_time_signal), num_blocks

def add_cyclic_prefix(signal, num_blocks, n_fft, cp_ratio):
    """Añade el prefijo cíclico a cada bloque OFDM"""
    cp_len = int(n_fft * cp_ratio)
    signal_with_cp = []
    
    # Procesar bloque por bloque
    for i in range(num_blocks):
        block = signal[i*n_fft : (i+1)*n_fft]
        cp = block[-cp_len:] # Copiar el final
        signal_with_cp.extend(np.concatenate((cp, block)))
        
    return np.array(signal_with_cp), cp_len

def remove_cyclic_prefix(rx_signal, n_fft, cp_len):
    """Elimina el CP asumiendo sincronización perfecta"""
    block_len = n_fft + cp_len
    num_blocks = len(rx_signal) // block_len
    rx_no_cp = []
    
    for i in range(num_blocks):
        # Extraer bloque completo con CP
        full_block = rx_signal[i*block_len : (i+1)*block_len]
        # Quedarse solo con la parte útil (quitar CP del inicio)
        useful_part = full_block[cp_len:]
        rx_no_cp.extend(useful_part)
        
    return np.array(rx_no_cp)

def demodulate_ofdm(rx_time_signal, n_fft, nc):
    """Aplica FFT para recuperar símbolos en frecuencia"""
    num_blocks = len(rx_time_signal) // n_fft
    rx_symbols_freq = []
    
    for i in range(num_blocks):
        time_block = rx_time_signal[i*n_fft : (i+1)*n_fft]
        fft_out = np.fft.fft(time_block) / np.sqrt(n_fft) # Normalización inversa
        
        # Extraer las subportadoras de datos (misma lógica que en Tx)
        data_subcarriers = fft_out[1:nc+1]
        rx_symbols_freq.extend(data_subcarriers)
        
    return np.array(rx_symbols_freq)

def equalize_channel(rx_freq_symbols, h_impulse_response, n_fft, nc):
    """
    Ecualizador Zero-Forcing (1-tap).
    Divide la señal recibida por la respuesta del canal en frecuencia.
    
    rx_freq_symbols: Símbolos recibidos tras la FFT
    h_impulse_response: Respuesta al impulso del canal (h) que devolvió el módulo Channel
    """
    # 1. Obtener la respuesta en Frecuencia del canal (H)
    # La FFT de h debe ser del mismo tamaño que la FFT de la señal (N_FFT)
    H_freq = np.fft.fft(h_impulse_response, n_fft)
    
    # 2. Extraer los valores de H correspondientes a las subportadoras de datos
    # (Debemos usar los mismos índices que usamos en modulate_ofdm)
    H_data = H_freq[1:nc+1] 
    
    # 3. Ecualización: Y = X * H + N  ==>  X_est = Y / H
    # Procesamos bloque a bloque porque H se aplica a cada bloque OFDM
    num_blocks = len(rx_freq_symbols) // nc
    equalized_symbols = []
    
    # Evitar división por cero
    threshold = 1e-10
    H_data[np.abs(H_data) < threshold] = threshold
    
    for i in range(num_blocks):
        block_y = rx_freq_symbols[i*nc : (i+1)*nc]
        # División elemento a elemento
        block_x_est = block_y / H_data
        equalized_symbols.extend(block_x_est)
        
    return np.array(equalized_symbols)


def apply_sfbc_encoding(symbols):
    """
    Codificación Alamouti/SFBC para 2 antenas.
    Entrada: Array de símbolos (debe ser par).
    Salida: Tuple (symbols_ant1, symbols_ant2)
    """
    # Asegurar longitud par (padding si es necesario)
    if len(symbols) % 2 != 0:
        symbols = np.append(symbols, 0)
    
    # Separar en pares (s0, s1)
    s0 = symbols[0::2]
    s1 = symbols[1::2]
    
    # Construir vectores para cada antena
    # Antena 1: [s0, -s1*]
    ant1 = np.empty_like(symbols)
    ant1[0::2] = s0
    ant1[1::2] = -np.conj(s1)
    
    # Antena 2: [s1, s0*]
    ant2 = np.empty_like(symbols)
    ant2[0::2] = s1
    ant2[1::2] = np.conj(s0)
    
    return ant1, ant2


def decode_sfbc(rx_symbols, h1_freq, h2_freq, nc):
    """
    Decodifica la señal recibida usando las estimaciones de canal H1 y H2.
    """
    # 1. Extraer subportadoras de datos de los canales (Un solo bloque)
    H1_one_block = h1_freq[1:nc+1]
    H2_one_block = h2_freq[1:nc+1]
    
    # 2. Calcular cuántos bloques OFDM recibimos en total
    # rx_symbols contiene TODOS los bloques concatenados
    num_blocks = len(rx_symbols) // nc
    
    # 3. Repetir el canal H para que cubra todos los bloques
    H1_full = np.tile(H1_one_block, num_blocks)
    H2_full = np.tile(H2_one_block, num_blocks)
    
    # 4. Separar lo recibido y el canal en pares (r0, r1) y (H_pair)
    # Al hacer slicing [0::2] sobre H1_full, ahora sí tiene el mismo tamaño que r0
    r0 = rx_symbols[0::2]
    r1 = rx_symbols[1::2]
    
    H1_pair = H1_full[0::2] 
    H2_pair = H2_full[0::2]
    
    # 5. Denominador (Norma al cuadrado)
    norm_sq = (np.abs(H1_pair)**2 + np.abs(H2_pair)**2)
    # Evitar división por cero
    norm_sq[norm_sq == 0] = 1e-10
    
    # 6. Estimación (Fórmulas de Alamouti)
    # Ahora todas las matrices tienen el mismo tamaño
    s0_est = (np.conj(H1_pair) * r0 + H2_pair * np.conj(r1)) / norm_sq
    s1_est = (np.conj(H2_pair) * r0 - H1_pair * np.conj(r1)) / norm_sq
    
    # 7. Reconstruir el stream único
    # Preparamos el array de salida
    decoded = np.empty(len(rx_symbols), dtype=complex)
    decoded[0::2] = s0_est
    decoded[1::2] = s1_est
    
    return decoded