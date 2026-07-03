#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def calcola_metriche(soglia, mu_g, sig_g, mu_i, sig_i):
    """Calcola analiticamente FAR e FRR basandosi sulle distribuzioni gaussiane."""
    # FRR: Genuine rifiutati (area a destra della soglia)
    # Usiamo la funzione di ripartizione cumulativa (CDF) approssimata o esatta
    from scipy.stats import norm
    frr = (1 - norm.cdf(soglia, mu_g, sig_g)) * 100
    # FAR: Impostori accettati (area a sinistra della soglia)
    far = norm.cdf(soglia, mu_i, sig_i) * 100
    return far, frr

def main():
    # 1. Configurazione iniziale dei dati reali ricavati dall'ultimo run
    mu_genuine = 0.31
    sigma_genuine = 0.035
    mu_impostor = 0.44
    sigma_impostor = 0.025

    # Generazione dei punti per le curve di densità
    x = np.linspace(0.15, 0.55, 500)
    
    # Creazione della figura e del layout
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25) # Lascia spazio in basso per gli slider

    # Disegno iniziale delle curve
    from scipy.stats import norm
    line_genuine, = ax.plot(x, norm.pdf(x, mu_genuine, sigma_genuine), label='Genuine (Match Veri)', lw=2)
    line_impostor, = ax.plot(x, norm.pdf(x, mu_impostor, sigma_impostor), label='Impostori / Gemelli', lw=2)
    
    # Riempimento aree per estetica
    area_g = ax.fill_between(x, norm.pdf(x, mu_genuine, sigma_genuine), alpha=0.3)
    area_i = ax.fill_between(x, norm.pdf(x, mu_impostor, sigma_impostor), alpha=0.3)

    # Linea verticale della soglia di decisione (iniziale a 0.36)
    soglia_iniziale = 0.36
    linea_soglia = ax.axvline(x=soglia_iniziale, color='black', linestyle='--', lw=2, label='Soglia di Decisione')

    # Configurazione assi e testi
    ax.set_title("Analisi Biometrica Live: Impatto del Rumore e della Soglia", fontsize=14, pad=15)
    ax.set_xlabel("Hamming Distance (HD)")
    ax.set_ylabel("Densità di Probabilità")
    ax.legend(loc='upper left')
    ax.set_xlim(0.15, 0.55)
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3)

    # Aggiunta del testo dinamico per le metriche in tempo reale
    far_iniziale, frr_iniziale = calcola_metriche(soglia_iniziale, mu_genuine, sigma_genuine, mu_impostor, sigma_impostor)
    testo_metriche = ax.text(0.17, 15, f"FAR: {far_iniziale:.2f}%\nFRR: {frr_iniziale:.2f}%", 
                             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=11)

    # 2. Creazione degli Slider posizionati in basso
    ax_soglia = plt.axes([0.20, 0.12, 0.60, 0.03])
    ax_rumore = plt.axes([0.20, 0.05, 0.60, 0.03])

    slider_soglia = Slider(ax_soglia, 'Soglia (Threshold)', 0.20, 0.50, valinit=soglia_iniziale, valfmt='%.3f')
    slider_rumore = Slider(ax_rumore, 'Rumore / Variabilità', 0.0, 1.0, valinit=0.0, valfmt='%.1f')

    # 3. Funzione di aggiornamento dinamico chiamata al movimento degli slider
    def update(val):
        soglia = slider_soglia.val
        rumore = slider_rumore.val

        # Modifichiamo le distribuzioni in base al rumore ambientale
        # Il rumore allarga le curve (aumenta la varianza) e sposta i genuine a destra (peggiora il match)
        nuovo_sigma_g = sigma_genuine + (rumore * 0.03)
        nuovo_sigma_i = sigma_impostor + (rumore * 0.02)
        nuova_mu_g = mu_genuine + (rumore * 0.04)

        # Ricalcolo delle curve matematiche
        y_genuine = norm.pdf(x, nuova_mu_g, nuovo_sigma_g)
        y_impostor = norm.pdf(x, mu_impostor, nuovo_sigma_i)

        # Aggiornamento grafico delle linee
        line_genuine.set_ydata(y_genuine)
        line_impostor.set_ydata(y_impostor)

        # Aggiornamento della posizione della linea di soglia
        linea_soglia.set_xdata([soglia, soglia])

        # Ricalcolo e aggiornamento dei tassi di errore
        far, frr = calcola_metriche(soglia, nuova_mu_g, nuovo_sigma_g, mu_impostor, nuovo_sigma_i)
        testo_metriche.set_text(f"FAR: {far:.2f}%\nFRR: {frr:.2f}%")

        # Rinfresca la figura canvas
        fig.canvas.draw_idle()

    # Collegamento dei widget alla funzione di update
    slider_soglia.on_changed(update)
    slider_rumore.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()