import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

# Importamos el Controlador
from controller.simulation_mgr import OFDMSimulationManager

# Configuración inicial de apariencia
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Configuración de la Ventana Principal
        self.title("Simulador LTE OFDM")
        self.geometry("1100x750") # Un poco más alto para el nuevo botón

        # Inicializar el "Gerente"
        self.manager = OFDMSimulationManager()
        
        # Variable para almacenar la ruta de la imagen seleccionada
        self.selected_image_path = None
        
        # Mapeos
        self.bw_map = {"1.4 MHz": 1, "3 MHz": 2, "5 MHz": 3, "10 MHz": 4, "15 MHz": 5, "20 MHz": 6}
        self.cp_map = {"Normal (4.7µs)": 1, "Extendido (16.6µs)": 2}
        self.mod_map = {"QPSK": 1, "16-QAM": 2, "64-QAM": 3}

        # 2. Construcción de la Interfaz
        self.setup_ui()

    def setup_ui(self):
        """Disposición de elementos visuales (Grid Layout)"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL LATERAL (CONTROLES) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(20, weight=1) # Empujar status al fondo

        # Título
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="LTE SIMULATOR", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Grupo 1: Física
        self.lbl_phys = ctk.CTkLabel(self.sidebar_frame, text="Parámetros Físicos:", anchor="w")
        self.lbl_phys.grid(row=1, column=0, padx=20, pady=(10, 0))
        
        self.option_bw = ctk.CTkOptionMenu(self.sidebar_frame, values=list(self.bw_map.keys()))
        self.option_bw.grid(row=2, column=0, padx=20, pady=5)
        self.option_bw.set("10 MHz")

        self.option_cp = ctk.CTkOptionMenu(self.sidebar_frame, values=list(self.cp_map.keys()))
        self.option_cp.grid(row=3, column=0, padx=20, pady=5)
        self.option_cp.set("Normal (4.7µs)")

        # Grupo 2: Enlace y Canal
        self.lbl_link = ctk.CTkLabel(self.sidebar_frame, text="Modulación:", anchor="w")
        self.lbl_link.grid(row=4, column=0, padx=20, pady=(20, 0))

        self.option_mod = ctk.CTkOptionMenu(self.sidebar_frame, values=list(self.mod_map.keys()))
        self.option_mod.grid(row=5, column=0, padx=20, pady=5)
        self.option_mod.set("16-QAM")

        # Slider SNR eliminado (ya no se usa en modo batch)

        self.lbl_paths = ctk.CTkLabel(self.sidebar_frame, text="Caminos (Multipath): 1")
        self.lbl_paths.grid(row=6, column=0, padx=20, pady=(10,0))
        self.slider_paths = ctk.CTkSlider(self.sidebar_frame, from_=1, to=5, number_of_steps=4, command=self.update_paths_label)
        self.slider_paths.grid(row=7, column=0, padx=20, pady=5)
        self.slider_paths.set(1)

        self.switch_sfbc_var = ctk.StringVar(value="off")
        self.switch_sfbc = ctk.CTkSwitch(self.sidebar_frame, 
                                         text="Activar SFBC (MISO)", 
                                         command=self.update_sfbc_label,
                                         variable=self.switch_sfbc_var, 
                                         onvalue="on", offvalue="off")
        self.switch_sfbc.grid(row=8, column=0, padx=20, pady=(20, 0))

        # Grupo 3: Selección de Archivo
        self.lbl_source = ctk.CTkLabel(self.sidebar_frame, text="Fuente de Datos:", anchor="w")
        self.lbl_source.grid(row=9, column=0, padx=20, pady=(20, 0))

        self.btn_select_file = ctk.CTkButton(self.sidebar_frame, text="Seleccionar Imagen...", 
                                             fg_color="#4B4B4B", hover_color="#5B5B5B", 
                                             command=self.select_file)
        self.btn_select_file.grid(row=10, column=0, padx=20, pady=5)

        self.lbl_filename = ctk.CTkLabel(self.sidebar_frame, text="[Ningún archivo]", font=("Arial", 11), text_color="gray")
        self.lbl_filename.grid(row=11, column=0, padx=20, pady=0)

        # Grupo 4: Botones de Acción
        self.btn_run_img = ctk.CTkButton(self.sidebar_frame, text="TRANSMITIR IMAGEN", 
                                         fg_color="#1f538d", hover_color="#14375e",
                                         command=self.action_run_image)
        self.btn_run_img.grid(row=12, column=0, padx=20, pady=(20, 10))

        self.btn_run_ber = ctk.CTkButton(self.sidebar_frame, text="GENERAR CURVA BER", 
                                         fg_color="transparent", border_width=2, 
                                         command=self.action_plot_ber)
        self.btn_run_ber.grid(row=13, column=0, padx=20, pady=10)

        '''self.btn_run_papr = ctk.CTkButton(self.sidebar_frame, text="ANALIZAR PAPR", 
                                          fg_color="transparent", border_width=2, 
                                          command=self.action_plot_papr)
        self.btn_run_papr.grid(row=14, column=0, padx=20, pady=10)'''
        
        # --- CORRECCIÓN AQUÍ: Label de Estado movido al Sidebar ---
        self.lbl_status = ctk.CTkLabel(self.sidebar_frame, 
                                       text="Estado: Esperando", 
                                       text_color="gray",
                                       wraplength=200)
        # Usamos row=20 para asegurar que quede al final
        self.lbl_status.grid(row=20, column=0, padx=20, pady=20)


        # --- PANEL CENTRAL (VISUALIZACIÓN) ---
        self.tabview = ctk.CTkTabview(self, width=800)
        self.tabview.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        
        self.tab_img = self.tabview.add("Visualización Imagen")
        self.tab_ber = self.tabview.add("Análisis BER")
        #self.tab_papr = self.tabview.add("Análisis PAPR")

        # Configuración Tab Imagen 
        # NOTA: Ya no necesitamos crear labels aquí (lbl_tx_img, etc) porque 
        # la función action_run_image limpiará todo para poner la gráfica Matplotlib.
        self.tab_img.grid_columnconfigure(0, weight=1)


    # --- EVENTOS Y LÓGICA DE INTERFAZ ---

    def update_snr_label(self, value):
        self.lbl_snr.configure(text=f"SNR: {int(value)} dB")

    def update_paths_label(self, value):
        self.lbl_paths.configure(text=f"Caminos (Multipath): {int(value)}")

    def select_file(self):
        """Abre cuadro de diálogo para seleccionar imagen"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.selected_image_path = file_path
            # Mostrar solo el nombre del archivo para no saturar la UI
            filename = os.path.basename(file_path)
            if len(filename) > 25:
                filename = filename[:22] + "..."
            self.lbl_filename.configure(text=filename, text_color="#30D760") # Verde si hay archivo
            self.lbl_status.configure(text="Imagen cargada. Lista para transmitir.", text_color="white")

    def action_run_image(self):
        """Ejecuta la Comparativa Visual (Grilla 2x2)"""
        if not self.selected_image_path:
            messagebox.showwarning("Atención", "Selecciona una imagen primero.")
            return

        # Recolectar Inputs
        bw_idx = self.bw_map[self.option_bw.get()]
        prof_idx = self.cp_map[self.option_cp.get()]
        mod_idx = self.mod_map[self.option_mod.get()]
        paths = int(self.slider_paths.get()) # ¡Este SÍ lo usamos!
        
        # Como borramos el slider SNR, ya no lo leemos aquí.
        
        use_sfbc = (self.switch_sfbc_var.get() == "on")
        tech_str = "SFBC (MISO)" if use_sfbc else "SISO"

        self.lbl_status.configure(text=f"Procesando lote de imágenes ({tech_str})...")
        self.update()

        try:
            # Llamar al Manager (Ahora devuelve 5 simulaciones)
            batch_data = self.manager.run_comparison_batch(
                self.selected_image_path, bw_idx, prof_idx, mod_idx, paths, use_sfbc
            )

            if not batch_data:
                messagebox.showerror("Error", "Falló la simulación por lotes.")
                return

            # --- VISUALIZACIÓN EN GRILLA 2x3 (6 IMÁGENES) ---
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # Limpiar frame anterior
            for widget in self.tab_img.winfo_children():
                widget.destroy()

            # 1. Crear Figura más ancha (12x7 pulgadas)
            fig = plt.Figure(figsize=(12, 7), dpi=100)
            fig.patch.set_facecolor('#2b2b2b') 
            
            # 2. Plotear ORIGINAL (Posición 1)
            # add_subplot(Filas, Columnas, Índice) -> (2, 3, 1)
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(batch_data['original'], cmap='gray', vmin=0, vmax=255)
            ax1.set_title("ORIGINAL (Tx)", color='white', fontsize=10, fontweight='bold')
            ax1.axis('off')

            # 3. Plotear las 5 SIMULACIONES (Posiciones 2 a 6)
            sims = batch_data['simulations']
            
            # Iteramos automáticamente. 
            # i va de 0 a 4. El subplot debe ir de 2 a 6.
            for i, sim in enumerate(sims):
                plot_idx = i + 2  # Posición en la grilla (2, 3, 4, 5, 6)
                ax = fig.add_subplot(2, 3, plot_idx)
                self._plot_subplot(ax, sim)

            fig.tight_layout()

            # Insertar en GUI
            canvas = FigureCanvasTkAgg(fig, master=self.tab_img)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill='both', padx=5, pady=5)
            
            self.lbl_status.configure(text=f"Comparativa 6 pasos completada. Modo: {tech_str}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_status.configure(text="Error en simulación")
            messagebox.showerror("Error Crítico", str(e))

    def _plot_subplot(self, ax, sim_data):
        """Función auxiliar para no repetir código de plot"""
        ax.imshow(sim_data['rx_image'], cmap='gray', vmin=0, vmax=255)
        title = f"SNR: {sim_data['snr']} dB\nBER: {sim_data['ber']:.4f}"
        
        # Usamos colores para indicar calidad
        color_title = '#ff5555' # Rojo (Malo)
        if sim_data['ber'] < 0.1: color_title = '#ffb86c' # Naranja (Regular)
        if sim_data['ber'] < 0.01: color_title = '#30D760' # Verde (Bueno)

        ax.set_title(title, color=color_title, fontsize=9, fontweight='bold')
        ax.axis('off')

    def action_plot_ber(self):
        """Genera curvas comparativas de BER"""
        
        # 1. Recolectar parámetros
        bw_idx = self.bw_map[self.option_bw.get()]
        prof_idx = self.cp_map[self.option_cp.get()]
        mod_idx = self.mod_map[self.option_mod.get()]
        paths = int(self.slider_paths.get())
        
        # 2. Determinar Modo de Comparación
        use_sfbc = (self.switch_sfbc_var.get() == "on")
        
        if use_sfbc:
            compare_mode = "DIVERSITY"
            status_msg = "Generando comparación: SISO vs SFBC..."
        else:
            compare_mode = "MODULATIONS"
            status_msg = "Generando comparación: QPSK vs 16QAM vs 64QAM..."

        self.lbl_status.configure(text=status_msg)
        self.update()

        try:
            # 3. Llamar al Manager
            # Nota: Usamos la nueva función 'run_ber_comparison_logic'
            curves_data = self.manager.run_ber_comparison_logic(
                bw_idx, prof_idx, mod_idx, paths, compare_mode
            )

            # 4. Graficar
            self._plot_ber_curves(curves_data)
            
            self.lbl_status.configure(text="Curvas BER generadas exitosamente.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_status.configure(text="Error generando BER")
            messagebox.showerror("Error", str(e))

    def _plot_ber_curves(self, curves_data):
        """Función auxiliar para dibujar múltiples curvas con estilos personalizados"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Limpiar Tab
        for widget in self.tab_ber.winfo_children():
            widget.destroy()

        # Crear Figura
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        # Iterar sobre las curvas recibidas
        for curve in curves_data:
            ax.semilogy(curve['x'], curve['y'], 
                        marker=curve.get('marker', 'o'),    # Usa el marcador del dict o 'o' por defecto
                        linestyle=curve.get('linestyle', '-'), # Usa el estilo del dict o '-'
                        label=curve['label'], 
                        color=curve['color'], 
                        alpha=curve.get('alpha', 1.0),      # Transparencia opcional
                        linewidth=2,
                        markersize=5)

        # Configuración Estética
        ax.set_title("Comparativa Total: Modulación vs Diversidad", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]", color='white')
        ax.set_ylabel("BER (Escala Logarítmica)", color='white')
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Grid y Límites
        ax.grid(True, which="both", color='#444444', linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0.00001, top=1.0) 

        # Leyenda (Importante para entender las 6 curvas)
        # La colocamos fuera o en la mejor posición posible
        legend = ax.legend(frameon=True, facecolor='#2b2b2b', edgecolor='white', fontsize=8, loc='best')
        for text in legend.get_texts():
            text.set_color("white")

        fig.tight_layout()

        # Insertar en GUI
        canvas = FigureCanvasTkAgg(fig, master=self.tab_ber)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)

    def action_plot_papr(self):
        """Genera y muestra gráfica PAPR con la IMAGEN"""
        if not self.selected_image_path:
            messagebox.showwarning("Falta Imagen", "Selecciona una imagen para analizar su PAPR.")
            return

        bw_idx = self.bw_map[self.option_bw.get()]
        prof_idx = self.cp_map[self.option_cp.get()]
        mod_idx = self.mod_map[self.option_mod.get()]

        self.lbl_status.configure(text="Calculando PAPR de la imagen...")
        self.update()
        
        try:
            # --- CAMBIO: Pasamos self.selected_image_path ---
            thresholds, ccdf = self.manager.calculate_papr_distribution(
                self.selected_image_path, bw_idx, prof_idx, mod_idx
            )

            self.embed_plot(self.tab_papr, thresholds, ccdf, 
                           "CCDF de PAPR (Datos de Imagen)", "Umbral de Potencia (dB)", "Probabilidad (PAPR > Umbral)", log_y=True)
            self.lbl_status.configure(text="Gráfica PAPR generada.")
            self.tabview.set("Análisis PAPR")
            
        except Exception as e:
             self.lbl_status.configure(text="Error en cálculo PAPR")
             messagebox.showerror("Error", str(e))

    def embed_plot(self, parent_frame, x_data, y_data, title, xlabel, ylabel, log_y=False):
        """Función auxiliar para incrustar gráficos Matplotlib en CustomTkinter"""
        # Limpiar gráfico anterior si existe
        for widget in parent_frame.winfo_children():
            widget.destroy()

        # Crear Figura
        fig = Figure(figsize=(6, 4), dpi=100)
        # Fondo oscuro para coincidir con la app
        fig.patch.set_facecolor('#2b2b2b') 
        
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.plot(x_data, y_data, marker='o', color='#30D760', linewidth=2) # Verde SPTF
        
        ax.set_title(title, color='white', fontsize=12)
        ax.set_xlabel(xlabel, color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, color='#444444', linestyle='--')

        if log_y:
            ax.set_yscale('log')
            # Ajuste dinámico de límites para que se vea bien
            try:
                min_val = min(filter(lambda x: x > 0, y_data)) if any(y > 0 for y in y_data) else 1e-5
                ax.set_ylim(bottom=min_val*0.1, top=1.1)
            except:
                pass

        # Canvas de Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    
    def update_sfbc_label(self):
        """Callback opcional para ver en consola el cambio"""
        estado = self.switch_sfbc_var.get()
        print(f"Modo SFBC cambiado a: {estado}")