import numpy as np
import traceback
# Importamos el NÚCLEO (La física pura)
from core import config, utils, ofdm_ops, channel

class OFDMSimulationManager:
    """
    Recibe parámetros de la GUI, coordina los cálculos matemáticos del Core
    y devuelve resultados limpios para visualizar.
    """
    
    def __init__(self):
        
        pass

    def run_image_transmission(self, image_path, bw_idx, profile_idx, mod_type, snr_db, num_paths, use_sfbc=False):
        """
        Ejecuta la cadena completa: Tx -> Canal -> Rx
        Soporta modo SISO (Normal) y MISO (SFBC/Alamouti) mediante el flag use_sfbc.
        """
        try:
            # --- PASO 0: Configuración y Física ---
            n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
            
            # --- PASO 1: Transmisor (Tx) - Procesamiento de Bits ---
            # 1.1 Cargar imagen y convertir a bits
            img_size = 250
            tx_bits_raw, tx_img_matrix = utils.image_to_bits(image_path, img_size)
            
            # --- APLICAR SCRAMBLING ---
            tx_bits = utils.apply_scrambling(tx_bits_raw)
            
            # 1.2 Mapeo de Bits a Símbolos (Constelación)
            tx_symbols = utils.map_bits_to_symbols(tx_bits, mod_type)
            
            # =================================================================
            # DIVERGENCIA: MODO SFBC (2 Antenas) vs MODO NORMAL (1 Antena)
            # =================================================================
            
            if use_sfbc:
                # --- RAMA SFBC (MISO) ---
                
                # 1. Codificación Alamouti (Split en 2 antenas)
                sym_ant1, sym_ant2 = ofdm_ops.apply_sfbc_encoding(tx_symbols)
                
                # 2. Modulación OFDM (Doble IFFT)
                time_sig1, num_blocks = ofdm_ops.modulate_ofdm(sym_ant1, n_fft, nc)
                time_sig2, _          = ofdm_ops.modulate_ofdm(sym_ant2, n_fft, nc)
                
                # 3. Prefijo Cíclico (Doble CP)
                tx_cp1, cp_len = ofdm_ops.add_cyclic_prefix(time_sig1, num_blocks, n_fft, cp_ratio)
                tx_cp2, _      = ofdm_ops.add_cyclic_prefix(time_sig2, num_blocks, n_fft, cp_ratio)
                
                # 4. Canal MISO (2 Transmisores -> 1 Receptor)
                # Retorna la señal sumada y los dos canales (h1, h2)
                rx_signal_cp, h1, h2 = channel.apply_miso_rayleigh(tx_cp1, tx_cp2, snr_db, num_taps=num_paths)
                
            else:
                # --- RAMA SISO (Normal) ---
                
                # 1. Modulación OFDM (Una IFFT)
                ofdm_time_signal, num_blocks = ofdm_ops.modulate_ofdm(tx_symbols, n_fft, nc)
                
                # 2. Prefijo Cíclico
                tx_signal_cp, cp_len = ofdm_ops.add_cyclic_prefix(ofdm_time_signal, num_blocks, n_fft, cp_ratio)
                
                # 3. Canal SISO (1 Transmisor -> 1 Receptor)
                rx_signal_cp, h_channel = channel.apply_rayleigh(tx_signal_cp, snr_db, num_taps=num_paths)

            # =================================================================
            # --- PASO 3: Receptor (Rx) - Procesamiento Común ---
            # =================================================================
            
            # 3.1 Quitar el Prefijo Cíclico (Sincronización perfecta asumida)
            rx_signal_no_cp = ofdm_ops.remove_cyclic_prefix(rx_signal_cp, n_fft, cp_len)
            
            # 3.2 Demodulación OFDM (FFT) -> Símbolos Distorsionados
            rx_symbols_distorted = ofdm_ops.demodulate_ofdm(rx_signal_no_cp, n_fft, nc)
            
            # --- ECUALIZACIÓN / DECODIFICACIÓN ---
            
            if use_sfbc:
                # MODO SFBC: Necesitamos las respuestas en frecuencia de h1 y h2
                H1_freq = np.fft.fft(h1, n_fft)
                H2_freq = np.fft.fft(h2, n_fft)
                
                # Decodificador Alamouti
                rx_symbols_equalized = ofdm_ops.decode_sfbc(rx_symbols_distorted, H1_freq, H2_freq, nc)
            else:
                # MODO SISO: Ecualizador Zero-Forcing simple (1 tap)
                rx_symbols_equalized = ofdm_ops.equalize_channel(rx_symbols_distorted, h_channel, n_fft, nc)
            
            # 3.4 Demodulación Digital (Símbolos -> Bits Scrambled)
            rx_bits_scrambled = utils.demap_symbols_to_bits(rx_symbols_equalized, mod_type)
            
            # --- DESCRAMBLING ---
            valid_len = len(tx_bits_raw)
            rx_bits_scrambled = rx_bits_scrambled[:valid_len]
            rx_bits = utils.apply_scrambling(rx_bits_scrambled)
            
            # --- PASO 4: Métricas y Reconstrucción ---
            
            # 4.1 Calcular BER
            bit_errors = np.sum(tx_bits_raw != rx_bits)
            ber = bit_errors / valid_len
            
            # 4.2 Reconstruir Imagen
            rx_img_matrix = utils.bits_to_image(rx_bits, img_size)
            
            # Info string
            mod_str = config.MOD_CONSTELLATIONS[mod_type] if isinstance(config.MOD_CONSTELLATIONS[mod_type], str) else 'Custom'
            tech_str = "SFBC (MISO)" if use_sfbc else "SISO"

            return {
                "success": True,
                "tx_image": tx_img_matrix,
                "rx_image": rx_img_matrix,
                "ber": ber,
                "snr": snr_db,
                "info": f"BER: {ber:.5f} | {tech_str} | {mod_str}"
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def calculate_ber_curve(self, image_path, bw_idx, profile_idx, mod_type, num_paths, use_sfbc=False): # <--- 1. Nuevo parámetro
        """
        Calcula la curva BER usando los bits de la IMAGEN real.
        Soporta modo SISO y modo SFBC (Diversidad).
        """
        snr_range = np.linspace(0, 30, 10) 
        ber_values = []
        
        # Cargar bits de la imagen real ---
        # Usamos un tamaño menor para que la simulación de la curva sea rápida
        img_size = 100 
        
        tx_bits_raw, _ = utils.image_to_bits(image_path, img_size)
        
        # --- SCRAMBLING ---
        tx_bits = utils.apply_scrambling(tx_bits_raw)    
        
        # Parámetros físicos
        n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
        
        # Pre-modular la imagen (Mapeo a Símbolos)
        tx_syms = utils.map_bits_to_symbols(tx_bits, mod_type)
        
        # Iterar sobre las SNRs
        for snr in snr_range:
            
            # 2. BIFURCACIÓN DE LÓGICA (SISO vs SFBC)
            if use_sfbc:
                # === CAMINO SFBC (2 Antenas Tx -> 1 Rx) ===
                
                # A. Codificación Alamouti (Dividir en pares para 2 antenas)
                # Debes haber agregado esta función en ofdm_ops.py
                sym_ant1, sym_ant2 = ofdm_ops.apply_sfbc_encoding(tx_syms) 

                # B. Modulación OFDM (Doble rama)
                time_sig1, n_blks = ofdm_ops.modulate_ofdm(sym_ant1, n_fft, nc)
                time_sig2, _      = ofdm_ops.modulate_ofdm(sym_ant2, n_fft, nc)
                
                # C. Prefijo Cíclico (Doble CP)
                tx_cp1, cp_len = ofdm_ops.add_cyclic_prefix(time_sig1, n_blks, n_fft, cp_ratio)
                tx_cp2, _      = ofdm_ops.add_cyclic_prefix(time_sig2, n_blks, n_fft, cp_ratio)
                
                # D. Canal MISO (Simula h1 y h2, y suma las señales)
                # Debes haber agregado esta función en channel.py
                rx_cp, h1, h2 = channel.apply_miso_rayleigh(tx_cp1, tx_cp2, snr, num_taps=num_paths)
                
                # E. Recepción Común (Quitar CP y FFT)
                rx_no_cp = ofdm_ops.remove_cyclic_prefix(rx_cp, n_fft, cp_len)
                rx_distorted = ofdm_ops.demodulate_ofdm(rx_no_cp, n_fft, nc)
                
                # F. Decodificación SFBC (Alamouti)
                # Necesitamos la respuesta en frecuencia de ambos canales
                H1_freq = np.fft.fft(h1, n_fft)
                H2_freq = np.fft.fft(h2, n_fft)
                
                # Debes haber agregado esta función en ofdm_ops.py
                rx_eq = ofdm_ops.decode_sfbc(rx_distorted, H1_freq, H2_freq, nc)

            else:
                # === CAMINO SISO (Tu código original) ===
                
                # A. Modulación
                ofdm_sig, n_blks = ofdm_ops.modulate_ofdm(tx_syms, n_fft, nc)
                
                # B. CP
                tx_cp, cp_len = ofdm_ops.add_cyclic_prefix(ofdm_sig, n_blks, n_fft, cp_ratio)
                
                # C. Canal SISO
                rx_cp, h = channel.apply_rayleigh(tx_cp, snr, num_taps=num_paths)
                
                # D. Recepción y Ecualización Simple
                rx_no_cp = ofdm_ops.remove_cyclic_prefix(rx_cp, n_fft, cp_len)
                rx_syms = ofdm_ops.demodulate_ofdm(rx_no_cp, n_fft, nc)
                rx_eq = ofdm_ops.equalize_channel(rx_syms, h, n_fft, nc)

            # --- RECUPERACIÓN DE BITS (Común para ambos) ---
            rx_bits_scrambled = utils.demap_symbols_to_bits(rx_eq, mod_type)
            
            # 3. DESCRAMBLING
            # Recortar longitud antes de descrambling para evitar errores de tamaño
            valid_len = len(tx_bits_raw)
            rx_bits_scrambled = rx_bits_scrambled[:valid_len]
            
            rx_bits = utils.apply_scrambling(rx_bits_scrambled)

            # 4. Cálculo de BER
            ber = np.sum(tx_bits_raw != rx_bits) / valid_len
            ber_values.append(ber)
            
        return snr_range, ber_values

    def calculate_papr_distribution(self, image_path, bw_idx, profile_idx, mod_type):
        """
        Calcula la CCDF del PAPR usando los bloques OFDM generados por la IMAGEN.
        """
        n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
        
        # Datos de la imagen ---
        img_size = 250
        tx_bits_raw, _ = utils.image_to_bits(image_path, img_size)
        tx_bits = utils.apply_scrambling(tx_bits_raw)
        
        # 1. Mapear y Modular toda la imagen
        syms = utils.map_bits_to_symbols(tx_bits, mod_type)
        # Esto nos devuelve la señal completa en el tiempo y cuántos bloques ocupó
        time_signal, num_blocks = ofdm_ops.modulate_ofdm(syms, n_fft, nc)
        
        papr_values = []
        
        # 2. Calcular PAPR bloque por bloque de la imagen
        # La señal 'time_signal' es una concatenación de todos los bloques IFFT
        for i in range(num_blocks):
            # Extraer el bloque i-ésimo
            block = time_signal[i*n_fft : (i+1)*n_fft]
            
            power = np.abs(block)**2
            peak_pwr = np.max(power)
            avg_pwr = np.mean(power)
            
            if avg_pwr > 0:
                papr_val = 10 * np.log10(peak_pwr / avg_pwr)
                papr_values.append(papr_val)
        
        # 3. Crear curva CCDF (Igual que antes)
        thresholds = np.linspace(0, 12, 100) # dB
        ccdf = []
        papr_array = np.array(papr_values)
        
        for x in thresholds:
            if len(papr_array) > 0:
                prob = np.sum(papr_array > x) / len(papr_array)
            else:
                prob = 0
            ccdf.append(prob)
            
        return thresholds, ccdf