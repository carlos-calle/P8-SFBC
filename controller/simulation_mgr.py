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
            
            # MODO SFBC (2 Antenas) vs MODO NORMAL (1 Antena)
            
            if use_sfbc:
                # --- RAMA SFBC (MISO) ---
                
                # 1. Codificación Alamouti 
                sym_ant1, sym_ant2 = ofdm_ops.apply_sfbc_encoding(tx_symbols)
                
                # 2. Modulación OFDM
                time_sig1, num_blocks = ofdm_ops.modulate_ofdm(sym_ant1, n_fft, nc)
                time_sig2, _          = ofdm_ops.modulate_ofdm(sym_ant2, n_fft, nc)
                
                # 3. Prefijo Cíclico
                tx_cp1, cp_len = ofdm_ops.add_cyclic_prefix(time_sig1, num_blocks, n_fft, cp_ratio)
                tx_cp2, _      = ofdm_ops.add_cyclic_prefix(time_sig2, num_blocks, n_fft, cp_ratio)
                
                # 4. Canal MISO (2 Transmisores -> 1 Receptor)
                # Retorna la señal sumada y los dos canales (h1, h2)
                rx_signal_cp, h1, h2 = channel.apply_miso_rayleigh(tx_cp1, tx_cp2, snr_db, num_taps=num_paths)
                
            else:
                # --- RAMA SISO (Normal) ---
                
                # 1. Modulación OFDM 
                ofdm_time_signal, num_blocks = ofdm_ops.modulate_ofdm(tx_symbols, n_fft, nc)
                
                # 2. Prefijo Cíclico
                tx_signal_cp, cp_len = ofdm_ops.add_cyclic_prefix(ofdm_time_signal, num_blocks, n_fft, cp_ratio)
                
                # 3. Canal SISO (1 Transmisor -> 1 Receptor)
                rx_signal_cp, h_channel = channel.apply_rayleigh(tx_signal_cp, snr_db, num_taps=num_paths)

            # --- PASO 3: Receptor (Rx) - Procesamiento Común ---
            
            # 3.1 Quitar el Prefijo Cíclico
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
        img_size = 250 
        
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
                sym_ant1, sym_ant2 = ofdm_ops.apply_sfbc_encoding(tx_syms) 

                # B. Modulación OFDM (Doble rama)
                time_sig1, n_blks = ofdm_ops.modulate_ofdm(sym_ant1, n_fft, nc)
                time_sig2, _      = ofdm_ops.modulate_ofdm(sym_ant2, n_fft, nc)
                
                # C. Prefijo Cíclico (Doble CP)
                tx_cp1, cp_len = ofdm_ops.add_cyclic_prefix(time_sig1, n_blks, n_fft, cp_ratio)
                tx_cp2, _      = ofdm_ops.add_cyclic_prefix(time_sig2, n_blks, n_fft, cp_ratio)
                
                # D. Canal MISO (Simula h1 y h2, y suma las señales)
                rx_cp, h1, h2 = channel.apply_miso_rayleigh(tx_cp1, tx_cp2, snr, num_taps=num_paths)
                
                # E. Recepción Común (Quitar CP y FFT)
                rx_no_cp = ofdm_ops.remove_cyclic_prefix(rx_cp, n_fft, cp_len)
                rx_distorted = ofdm_ops.demodulate_ofdm(rx_no_cp, n_fft, nc)
                
                # F. Decodificación SFBC (Alamouti)
                H1_freq = np.fft.fft(h1, n_fft)
                H2_freq = np.fft.fft(h2, n_fft)
                
                rx_eq = ofdm_ops.decode_sfbc(rx_distorted, H1_freq, H2_freq, nc)

            else:
                # === CAMINO SISO ===
                
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
    

    def run_comparison_batch(self, image_path, bw_idx, profile_idx, mod_type, num_paths, use_sfbc):
        """
        Ejecuta 3 simulaciones seguidas (Baja, Media, Alta SNR) para comparar.
        """
        snr_levels = [4, 8, 12, 16, 20]
        results = []
        
        tx_image = None # Para guardar la original una sola vez

        for snr in snr_levels:
            # Reutilizamos tu función existente
            res = self.run_image_transmission(
                image_path, bw_idx, profile_idx, mod_type, snr, num_paths, use_sfbc
            )
            
            if res['success']:
                if tx_image is None:
                    tx_image = res['tx_image']
                
                # Guardamos lo que nos interesa mostrar
                results.append({
                    'snr': snr,
                    'rx_image': res['rx_image'],
                    'ber': res['ber']
                })
            else:
                return None # Si falla algo, abortamos

        return {
            "original": tx_image,
            "simulations": results
        }
    

    def _calculate_single_curve(self, bw_idx, profile_idx, mod_type, num_paths, use_sfbc):
        """
        Calcula una curva BER promediando múltiples realizaciones de canal
        para eliminar los "picos" de mala suerte (Smooth Curves).
        """
        snr_range = range(0, 31, 3) 
        ber_values = []
        
        # Parámetros de Monte Carlo
        # Ajusta esto: Más iteraciones = Curva más suave, pero más lento.
        # 50 iteraciones con 10,000 bits suele ser mejor que 1 iteración con 500,000 bits.
        channel_iters = 100  
        bits_per_iter = 10000 # Bits por cada "intento" de canal

        # Pre-cálculo de parámetros físicos
        n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)

        for snr in snr_range:
            total_errors = 0
            total_bits = 0
            
            # --- BUCLE DE PROMEDIO DE CANAL ---
            # Probamos 'channel_iters' escenarios distintos para este mismo SNR
            for _ in range(channel_iters):
                
                # 1. Generar bits nuevos para cada iteración
                tx_bits_raw = np.random.randint(0, 2, bits_per_iter)
                tx_bits = utils.apply_scrambling(tx_bits_raw)

                # 2. Modulación
                symbols = utils.map_bits_to_symbols(tx_bits, mod_type)
                
                # 3. Encoding y OFDM
                if use_sfbc:
                    if len(symbols) % 2 != 0: symbols = np.append(symbols, 0)
                    ant1, ant2 = ofdm_ops.apply_sfbc_encoding(symbols)
                    tx_sig1, n_blks = ofdm_ops.modulate_ofdm(ant1, n_fft, nc)
                    tx_sig2, _      = ofdm_ops.modulate_ofdm(ant2, n_fft, nc)
                    tx_cp1, cp_len = ofdm_ops.add_cyclic_prefix(tx_sig1, n_blks, n_fft, cp_ratio)
                    tx_cp2, _      = ofdm_ops.add_cyclic_prefix(tx_sig2, n_blks, n_fft, cp_ratio)
                else:
                    tx_sig, n_blks = ofdm_ops.modulate_ofdm(symbols, n_fft, nc)
                    tx_cp, cp_len  = ofdm_ops.add_cyclic_prefix(tx_sig, n_blks, n_fft, cp_ratio)

                # 4. Canal (AQUÍ ES DONDE SE GENERA EL "H" ALEATORIO NUEVO)
                if use_sfbc:
                    rx_cp, h1, h2 = channel.apply_miso_rayleigh(tx_cp1, tx_cp2, snr, num_paths)
                else:
                    rx_cp, h = channel.apply_rayleigh(tx_cp, snr, num_paths)

                # 5. Recepción
                rx_no_cp = ofdm_ops.remove_cyclic_prefix(rx_cp, n_fft, cp_len)
                rx_freq = ofdm_ops.demodulate_ofdm(rx_no_cp, n_fft, nc)

                if use_sfbc:
                    h1_freq = np.fft.fft(h1, n_fft)
                    h2_freq = np.fft.fft(h2, n_fft)
                    est_symbols = ofdm_ops.decode_sfbc(rx_freq, h1_freq, h2_freq, nc)
                else:
                    h_freq = np.fft.fft(h, n_fft)
                    est_symbols = ofdm_ops.equalize_channel(rx_freq, h, n_fft, nc)

                # 6. Demapeo y Conteo
                rx_bits_scrambled = utils.demap_symbols_to_bits(est_symbols, mod_type)
                rx_bits_scrambled = rx_bits_scrambled[:len(tx_bits_raw)] # Ajuste por padding
                rx_bits = utils.apply_scrambling(rx_bits_scrambled)
                
                # Acumulamos estadísticas
                errors = np.sum(tx_bits_raw != rx_bits)
                total_errors += errors
                total_bits += len(tx_bits_raw)

            # Fin del bucle de canales. Calculamos BER promedio para este SNR
            if total_bits > 0:
                avg_ber = total_errors / total_bits
            else:
                avg_ber = 0
            
            ber_values.append(avg_ber)

        return list(snr_range), ber_values

    def run_ber_comparison_logic(self, bw_idx, profile_idx, current_mod, num_paths, compare_mode):
        """
        Lógica principal de comparación.
        compare_mode: "MODULATIONS" (Si switch off) o "DIVERSITY" (Si switch on)
        """
        curves = [] 
        
        # Mapa de colores y nombres consistente
        colors = {1: "#30D760", 2: "#FFD700", 3: "#FF5555"} # Verde, Dorado, Rojo
        mod_names = {1: "QPSK", 2: "16-QAM", 3: "64-QAM"}

        if compare_mode == "MODULATIONS":
            # Caso 1: Switch Apagado -> Solo SISO para las 3 modulaciones
            for mod_idx in [1, 2, 3]:
                x, y = self._calculate_single_curve(bw_idx, profile_idx, mod_idx, num_paths, use_sfbc=False)
                curves.append({
                    'label': f"{mod_names[mod_idx]} (SISO)",
                    'x': x, 'y': y,
                    'color': colors[mod_idx],
                    'linestyle': '-',  # Línea normal
                    'marker': 'o'
                })

        elif compare_mode == "DIVERSITY":
            # Caso 2: Switch Encendido -> Comparativa TOTAL (6 Curvas)
            # Iteramos sobre todas las modulaciones para mostrar el panorama completo
            for mod_idx in [1, 2, 3]:
                name = mod_names[mod_idx]
                col = colors[mod_idx]

                # A. Curva SISO (Línea Punteada - "El pasado")
                x1, y1 = self._calculate_single_curve(bw_idx, profile_idx, mod_idx, num_paths, use_sfbc=False)
                curves.append({
                    'label': f"{name} - SISO",
                    'x': x1, 'y': y1,
                    'color': col,
                    'linestyle': '--', # Dashed
                    'marker': 'x',     # Marcador 'x' para diferenciar
                    'alpha': 0.7       # Un poco más suave
                })

                # B. Curva SFBC (Línea Sólida - "El futuro")
                x2, y2 = self._calculate_single_curve(bw_idx, profile_idx, mod_idx, num_paths, use_sfbc=True)
                curves.append({
                    'label': f"{name} - SFBC",
                    'x': x2, 'y': y2,
                    'color': col,
                    'linestyle': '-',  # Solid
                    'marker': 'o',     # Marcador círculo sólido
                    'alpha': 1.0       # Color pleno
                })

        return curves