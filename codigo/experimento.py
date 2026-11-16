"""Tarea 3 - Ajuste de parametros con diseno de experimentos.

Experimento factorial 3 x 3 para TSP con Recocido Simulado y movimientos 2-opt.
La implementacion conserva la estructura del Ejercicio 4 del repositorio del alumno.
"""

from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parents[1]
RESULTADOS = BASE / "resultados"
GRAFICAS = BASE / "graficas"
RESULTADOS.mkdir(exist_ok=True)
GRAFICAS.mkdir(exist_ok=True)

# Parametros que se mantienen constantes durante el experimento.
SEMILLA_R = 444
SEMILLA_INSTANCIAS = 2026
ITERACIONES = 200
REPLICAS = 10
TEMPERATURA_FINAL = 1e-3
TEMPERATURA_MINIMA = 0.10
TEMPERATURA_MAXIMA = 1.00
TAMANOS = [5, 10, 15]


def ciudades_aleatorias(n, rng):
    return rng.random((n, 2))


def matriz_distancias(p):
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)


def costo_tour(t, D):
    t = np.asarray(t)
    return float(D[t, np.roll(t, -1)].sum())


def recocido_simulado(D, tour0, iteraciones, T0, Tf, rng):
    """Recocido simulado con enfriamiento exponencial y vecindarios 2-opt."""
    n = len(D)
    tour = np.array(tour0, dtype=int)
    costo = costo_tour(tour, D)
    mejor = tour.copy()
    mejor_costo = costo

    for k in range(iteraciones):
        fraccion = k / iteraciones
        temperatura = T0 * (Tf / T0) ** fraccion

        i = int(rng.integers(1, n - 1))
        j = int(rng.integers(i + 1, n))
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % n]
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]

        if delta < 0 or rng.random() < np.exp(-delta / temperatura):
            tour[i:j + 1] = tour[i:j + 1][::-1].copy()
            costo += delta
            if costo < mejor_costo - 1e-12:
                mejor_costo = costo
                mejor = tour.copy()

    # Recalculo para evitar que el error numerico acumulado afecte el reporte.
    return mejor, costo_tour(mejor, D)


def obtener_niveles():
    r = random.Random(SEMILLA_R).random()
    fracciones = [0.3 * r, 0.3 * r + 0.3, 0.3 * r + 0.6]
    temperaturas = [TEMPERATURA_MINIMA + f * (TEMPERATURA_MAXIMA - TEMPERATURA_MINIMA)
                    for f in fracciones]
    return r, fracciones, temperaturas


def bootstrap_mediana(valores, rng, n_boot=10_000):
    valores = np.asarray(valores, dtype=float)
    muestras = rng.choice(valores, size=(n_boot, len(valores)), replace=True)
    medianas = np.median(muestras, axis=1)
    inferior, superior = np.quantile(medianas, [0.025, 0.975])
    return float(np.median(valores)), float(inferior), float(superior)


def ejecutar_experimento():
    r, fracciones, niveles = obtener_niveles()
    print(f"r fijo = {r:.12f}")
    print("Temperaturas:", [round(x, 6) for x in niveles])

    instancias = {}
    for n in TAMANOS:
        rng_instancia = np.random.default_rng(SEMILLA_INSTANCIAS + n)
        puntos = ciudades_aleatorias(n, rng_instancia)
        instancias[n] = matriz_distancias(puntos)
        pd.DataFrame(puntos, columns=["x", "y"]).to_csv(
            RESULTADOS / f"instancia_{n}_ciudades.csv", index_label="ciudad")

    registros = []
    inicio_total = time.perf_counter()
    for n in TAMANOS:
        D = instancias[n]
        for replica in range(1, REPLICAS + 1):
            for indice_nivel, (nombre_nivel, T0) in enumerate(zip(["Baja", "Media", "Alta"], niveles)):
                # Cada ejecucion tiene una semilla independiente y reproducible.
                semilla_corrida = 50_000 + n * 1_000 + replica * 10 + indice_nivel
                rng_corrida = np.random.default_rng(semilla_corrida)
                tour_inicial = rng_corrida.permutation(n)
                t0 = time.perf_counter()
                mejor_tour, mejor_costo = recocido_simulado(
                    D, tour_inicial, ITERACIONES, T0, TEMPERATURA_FINAL, rng_corrida)
                tiempo = time.perf_counter() - t0
                registros.append({
                    "tamano": n,
                    "temperatura_nivel": nombre_nivel,
                    "T0": T0,
                    "replica": replica,
                    "costo_final": mejor_costo,
                    "tiempo_segundos": tiempo,
                    "ruta": "-".join(map(str, mejor_tour.tolist())),
                })
        print(f"Terminado n={n}")

    datos = pd.DataFrame(registros)
    datos.to_csv(RESULTADOS / "resultados.csv", index=False)

    # ANOVA factorial de dos factores, ambos tratados como categoricos.
    modelo = smf.ols("costo_final ~ C(temperatura_nivel) * C(tamano)", data=datos).fit()
    tabla_anova = sm.stats.anova_lm(modelo, typ=2)
    tabla_anova.to_csv(RESULTADOS / "anova_dos_factores.csv")

    # Diagnosticos basicos de los residuos.
    shapiro = stats.shapiro(modelo.resid)
    grupos = [g["costo_final"].values for _, g in datos.groupby(["temperatura_nivel", "tamano"])]
    levene = stats.levene(*grupos, center="median")
    with open(RESULTADOS / "diagnosticos.txt", "w", encoding="utf-8") as f:
        f.write(f"Shapiro-Wilk: estadistico={shapiro.statistic:.6f}, p={shapiro.pvalue:.6f}\n")
        f.write(f"Levene (mediana): estadistico={levene.statistic:.6f}, p={levene.pvalue:.6f}\n")
        f.write("Estos diagnosticos se interpretan como revisiones aproximadas de normalidad y homogeneidad.\n")

    # Resumen descriptivo por celda. No se atribuye significancia a estas diferencias.
    resumen = []
    for (n, nivel), grupo in datos.groupby(["tamano", "temperatura_nivel"], sort=True):
        resumen.append({
            "tamano": n,
            "temperatura_nivel": nivel,
            "T0": float(grupo["T0"].iloc[0]),
            "media_costo": grupo["costo_final"].mean(),
            "desv_estandar": grupo["costo_final"].std(ddof=1),
            "mediana_costo": grupo["costo_final"].median(),
        })
    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(RESULTADOS / "resumen_celdas.csv", index=False)

    # La ANOVA clasica identifica el tamano como factor significativo.
    # Por ello, el bootstrap se aplica a la mediana del costo para cada tamano,
    # agrupando sus tres niveles de temperatura.
    rng_boot = np.random.default_rng(98765)
    bootstrap_tamano = []
    p_tamano = float(tabla_anova.loc["C(tamano)", "PR(>F)"])
    if p_tamano < 0.05:
        for n, grupo in datos.groupby("tamano", sort=True):
            mediana, li, ls = bootstrap_mediana(grupo["costo_final"], rng_boot)
            bootstrap_tamano.append({
                "tamano": n,
                "n_observaciones": len(grupo),
                "mediana_costo": mediana,
                "IC95_inferior": li,
                "IC95_superior": ls,
            })
    pd.DataFrame(bootstrap_tamano).to_csv(RESULTADOS / "bootstrap_por_tamano.csv", index=False)

    # Mejor nivel observado por tamano: menor media; es una seleccion descriptiva.
    mejores = (resumen_df.sort_values(["tamano", "media_costo"])
               .groupby("tamano", as_index=False).first())
    mejores.to_csv(RESULTADOS / "mejor_temperatura_por_tamano.csv", index=False)

    # Analisis de sensibilidad con errores estandar robustos HC3.
    tabla_hc3 = sm.stats.anova_lm(modelo, typ=2, robust="hc3")
    tabla_hc3.to_csv(RESULTADOS / "anova_sensibilidad_HC3.csv")

    # Grafica de efectos principales.
    orden = ["Baja", "Media", "Alta"]
    medias_temp = datos.groupby("temperatura_nivel")["costo_final"].mean().reindex(orden)
    medias_n = datos.groupby("tamano")["costo_final"].mean().reindex(TAMANOS)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(orden, medias_temp.values, marker="o")
    ax[0].set_title("Efecto principal de la temperatura")
    ax[0].set_xlabel("Nivel de temperatura inicial")
    ax[0].set_ylabel("Costo final medio")
    ax[1].plot([str(n) for n in TAMANOS], medias_n.values, marker="o")
    ax[1].set_title("Efecto principal del tamaño")
    ax[1].set_xlabel("Número de ciudades")
    ax[1].set_ylabel("Costo final medio")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS / "efectos_principales.png", dpi=180)
    plt.close(fig)

    # Grafica de interaccion.
    medias_interaccion = datos.groupby(["tamano", "temperatura_nivel"])["costo_final"].mean().unstack()
    medias_interaccion = medias_interaccion.reindex(index=TAMANOS, columns=orden)
    plt.figure(figsize=(7, 4.5))
    for n in TAMANOS:
        plt.plot(orden, medias_interaccion.loc[n], marker="o", label=f"n={n}")
    plt.xlabel("Nivel de temperatura inicial")
    plt.ylabel("Costo final medio")
    plt.title("Grafica de interaccion: temperatura y tamano")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICAS / "interaccion.png", dpi=180)
    plt.close()

    # Grafica de distribuciones por celda, util para revisar dispersion.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    datos["celda"] = datos["tamano"].astype(str) + " ciudades - " + datos["temperatura_nivel"]
    etiquetas = [f"{n}-{nivel[0]}" for n in TAMANOS for nivel in orden]
    series = [datos.loc[(datos.tamano == n) & (datos.temperatura_nivel == nivel), "costo_final"]
              for n in TAMANOS for nivel in orden]
    ax.boxplot(series, labels=etiquetas)
    ax.set_xlabel("Tamaño - nivel (B=baja, M=media, A=alta)")
    ax.set_ylabel("Costo final")
    ax.set_title("Distribucion de costos por combinacion experimental")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS / "distribucion_costos.png", dpi=180)
    plt.close(fig)

    # Informacion reproducible para el reporte.
    with open(RESULTADOS / "configuracion.txt", "w", encoding="utf-8") as f:
        f.write("Tarea 3 - Diseno factorial 3 x 3\n")
        f.write(f"Semilla para r: {SEMILLA_R}\n")
        f.write(f"r: {r:.12f}\n")
        f.write(f"Semilla de instancias: {SEMILLA_INSTANCIAS}\n")
        f.write(f"Temperatura minima: {TEMPERATURA_MINIMA}\n")
        f.write(f"Temperatura maxima: {TEMPERATURA_MAXIMA}\n")
        f.write(f"Fracciones del rango: {[float(x) for x in fracciones]}\n")
        f.write(f"Temperaturas T0: {[float(x) for x in niveles]}\n")
        f.write(f"Temperatura final: {TEMPERATURA_FINAL}\n")
        f.write(f"Iteraciones por corrida: {ITERACIONES}\n")
        f.write(f"Replicas por celda: {REPLICAS}\n")
        f.write("Semillas independientes por ejecucion: si\n")
        f.write(f"Total de corridas: {len(datos)}\n")
        f.write(f"Tiempo total (s): {time.perf_counter() - inicio_total:.3f}\n")

    print("\nANOVA")
    print(tabla_anova)
    print("\nMejores temperaturas por tamano (seleccion descriptiva)")
    print(mejores.to_string(index=False))
    print("\nANOVA robusta HC3 (analisis de sensibilidad)")
    print(tabla_hc3)
    print(f"\nCorridas realizadas: {len(datos)}")


if __name__ == "__main__":
    ejecutar_experimento()
