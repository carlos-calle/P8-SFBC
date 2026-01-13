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

    def run_image_transmission(self, image_path, bw_idx, profile_idx, mod_type, snr_db, num_paths):
        """
        Ejecuta la cadena completa: Tx -> Canal -> Rx
        """
        try:
            # --- PASO 0: Configuración y Física ---
            # Obtenemos los números reales desde los índices de la GUI
            n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
            
            # --- PASO 1: Transmisor (Tx) ---
            # 1.1 Cargar imagen y convertir a bits
            # Usamos un tamaño fijo razonable para simulación rápida
            img_size = 250
            tx_bits_raw, tx_img_matrix = utils.image_to_bits(image_path, img_size)
            
            # --- APLICAR SCRAMBLING ---
            tx_bits = utils.apply_scrambling(tx_bits_raw)
            
            # 1.2 Mapeo de Bits a Símbolos (Constelación)
            tx_symbols = utils.map_bits_to_symbols(tx_bits, mod_type)
            
            # 1.3 Modulación OFDM (Dominio Frec -> Tiempo)
            ofdm_time_signal, num_blocks = ofdm_ops.modulate_ofdm(tx_symbols, n_fft, nc)
            
            # 1.4 Inserción de Prefijo Cíclico (Protección contra ecos)
            tx_signal_cp, cp_len = ofdm_ops.add_cyclic_prefix(ofdm_time_signal, num_blocks, n_fft, cp_ratio)
            
            # --- PASO 2: Canal (El Aire) ---
            # Aplicamos desvanecimiento Rayleigh y Ruido AWGN
            # IMPORTANTE: El canal nos devuelve 'h' (respuesta al impulso) para poder ecualizar después
            rx_signal_cp, h_channel = channel.apply_rayleigh(tx_signal_cp, snr_db, num_taps=num_paths)
            
            # --- PASO 3: Receptor (Rx) ---
            # 3.1 Quitar el Prefijo Cíclico
            rx_signal_no_cp = ofdm_ops.remove_cyclic_prefix(rx_signal_cp, n_fft, cp_len)
            
            # 3.2 Demodulación OFDM (Dominio Tiempo -> Frecuencia)
            rx_symbols_distorted = ofdm_ops.demodulate_ofdm(rx_signal_no_cp, n_fft, nc)
            
            # 3.3 ECUALIZACIÓN 
            # Dividimos lo recibido por 'h' para cancelar la distorsión del canal
            rx_symbols_equalized = ofdm_ops.equalize_channel(rx_symbols_distorted, h_channel, n_fft, nc)
            
            # 3.4 Demodulación Digital (Símbolos -> Bits)
            rx_bits_scrambled = utils.demap_symbols_to_bits(rx_symbols_equalized, mod_type)
            
            # DESCRAMBLING (Recuperar orden original) ---
            # Ajustamos longitud primero
            valid_len = len(tx_bits_raw)
            rx_bits_scrambled = rx_bits_scrambled[:valid_len]
            
            # Aplicamos la misma función para revertir el XOR
            rx_bits = utils.apply_scrambling(rx_bits_scrambled)
            
            # --- PASO 4: Métricas y Reconstrucción ---
            # 4.1 Calcular BER (Bit Error Rate)
            # Ajustar longitudes por si el padding añadió bits extra
            bit_errors = np.sum(tx_bits_raw != rx_bits)
            ber = bit_errors / valid_len
            
            # 4.2 Reconstruir Imagen
            rx_img_matrix = utils.bits_to_image(rx_bits, img_size)
            
            # Empaquetar todo en un diccionario para la GUI
            return {
                "success": True,
                "tx_image": tx_img_matrix,
                "rx_image": rx_img_matrix,
                "ber": ber,
                "snr": snr_db,
                "info": f"BER: {ber:.5f} | Modo: {config.MOD_CONSTELLATIONS[mod_type] if isinstance(config.MOD_CONSTELLATIONS[mod_type], str) else 'Custom'}"
            }

        except Exception as e:
            # Si algo explota (ej. imagen no encontrada), devolvemos el error ordenado
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def calculate_ber_curve(self, image_path, bw_idx, profile_idx, mod_type, num_paths):
        """
        Calcula la curva BER usando los bits de la IMAGEN real.
        """
        snr_range = np.linspace(0, 30, 10) 
        ber_values = []
        
        # Cargar bits de la imagen real ---
        # Usamos el mismo tamaño que en la transmisión normal para coherencia
        img_size = 250 
        # Si la carga falla, el try-except de la UI lo capturará
        tx_bits_raw, _ = utils.image_to_bits(image_path, img_size)
        tx_bits = utils.apply_scrambling(tx_bits_raw)    
        
        # Parámetros físicos
        n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
        
        # Pre-modular la imagen
        tx_syms = utils.map_bits_to_symbols(tx_bits, mod_type)
        
        # Iterar sobre las SNRs
        for snr in snr_range:
            # 1. Modulación
            ofdm_sig, n_blks = ofdm_ops.modulate_ofdm(tx_syms, n_fft, nc)
            tx_cp, cp_len = ofdm_ops.add_cyclic_prefix(ofdm_sig, n_blks, n_fft, cp_ratio)
            
            # 2. Canal (Aplica ruido diferente en cada iteración según la SNR)
            rx_cp, h = channel.apply_rayleigh(tx_cp, snr, num_taps=num_paths)
            
            # 3. Recepción
            rx_no_cp = ofdm_ops.remove_cyclic_prefix(rx_cp, n_fft, cp_len)
            rx_syms = ofdm_ops.demodulate_ofdm(rx_no_cp, n_fft, nc)
            rx_eq = ofdm_ops.equalize_channel(rx_syms, h, n_fft, nc)
            rx_bits_scrambled = utils.demap_symbols_to_bits(rx_eq, mod_type)
            rx_bits_scrambled = rx_bits_scrambled[:len(tx_bits)]
            rx_bits = utils.apply_scrambling(rx_bits_scrambled)

            # 4. Cálculo de BER
            # Recortar bits de relleno si los hay
            valid_len = len(tx_bits_raw)
            if len(rx_bits) > valid_len:
                rx_bits = rx_bits[:valid_len]

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