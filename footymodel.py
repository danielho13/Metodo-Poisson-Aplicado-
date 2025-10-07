# --- Paso 1: Ingesta y limpieza ---------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import pandas as pd

COLUMNAS_REQUERIDAS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}

class ConjuntoPartidos:
    """
    Maneja la carga y limpieza de los CSV de una liga (football-data.co.uk).
    Expone un DataFrame estándar con columnas:
      - date (pd.Timestamp, dayfirst=True)
      - home (str), away (str)
      - home_goals (int), away_goals (int)
    Solo mantiene partidos JUGADOS (sin NaN en goles).
    """

    def __init__(self, df: pd.DataFrame, nombre_liga: str):
        self._df = df.sort_values("date").reset_index(drop=True)
        self._nombre_liga = nombre_liga

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def nombre_liga(self) -> str:
        return self._nombre_liga

    # -------------------------- carga/estandarización ------------------------ #
    @staticmethod
    def _estandarizar_columnas(df_bruto: pd.DataFrame) -> pd.DataFrame:
        # Renombrar a esquema común
        renombrar = {
            "Date": "date",
            "HomeTeam": "home",
            "AwayTeam": "away",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            # tolerancia a alias poco comunes
            "Home": "home",
            "Away": "away",
            "HG": "home_goals",
            "AG": "away_goals",
        }
        df = df_bruto.rename(columns={k: v for k, v in renombrar.items() if k in df_bruto.columns})

        faltantes = {"date", "home", "away", "home_goals", "away_goals"} - set(df.columns)
        if faltantes:
            raise ValueError(
                f"Faltan columnas requeridas: {faltantes}. "
                f"Se esperaban al menos: {COLUMNAS_REQUERIDAS}. Columnas detectadas: {list(df_bruto.columns)}"
            )

        # Fechas en formato dd/mm/yy (football-data): dayfirst=True
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        if df["date"].isna().any():
            ejemplos = df.loc[df["date"].isna()].head(5)
            raise ValueError(
                "No se pudieron parsear algunas fechas (esperado dd/mm/yy). "
                f"Ejemplos problemáticos:\n{ejemplos}"
            )

        # Goles como numérico
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")

        # Solo partidos jugados (ambos goles presentes)
        df = df.dropna(subset=["home_goals", "away_goals"]).copy()

        return df[["date", "home", "away", "home_goals", "away_goals"]]

    @staticmethod
    def _leer_csv(ruta_csv: Path) -> pd.DataFrame:
        """
        Lee un CSV con tolerancia a BOM (utf-8-sig) y tipos mixtos.
        """
        if not ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
        df_bruto = pd.read_csv(ruta_csv, encoding="utf-8-sig", low_memory=False)
        if not COLUMNAS_REQUERIDAS.issubset(set(df_bruto.columns)):
            raise ValueError(
                f"El archivo {ruta_csv.name} no tiene las columnas mínimas {COLUMNAS_REQUERIDAS}. "
                f"Columnas detectadas: {list(df_bruto.columns)}"
            )
        return ConjuntoPartidos._estandarizar_columnas(df_bruto)

    @classmethod
    def desde_liga(cls, ruta_base_datos: Path, nombre_liga: str) -> "ConjuntoPartidos":
        """
        Carga y concatena los CSV de una liga: temp_ant.csv y temp_act.csv (si existen).
        Estructura esperada:
          ruta_base_datos / <liga> / temp_ant.csv
          ruta_base_datos / <liga> / temp_act.csv
        """
        carpeta_liga = ruta_base_datos / nombre_liga
        if not carpeta_liga.exists():
            raise FileNotFoundError(f"No existe la carpeta de liga: {carpeta_liga}")

        rutas: List[Path] = []
        for nombre in ("temp_ant.csv", "temp_act.csv"):
            ruta = carpeta_liga / nombre
            if ruta.exists():
                rutas.append(ruta)

        if not rutas:
            raise FileNotFoundError(
                f"La liga '{nombre_liga}' no tiene CSVs 'temp_ant.csv' ni 'temp_act.csv' en {carpeta_liga}"
            )

        marcos = [cls._leer_csv(p) for p in rutas]
        df = pd.concat(marcos, ignore_index=True)
        return cls(df, nombre_liga)

    # ------------------------------- utilidades ------------------------------ #
    def listar_equipos(self) -> list[str]:
        """Lista única de equipos (home ∪ away) ordenada alfabéticamente."""
        equipos = pd.unique(pd.concat([self._df["home"], self._df["away"]], ignore_index=True))
        return sorted([str(e) for e in equipos])

    def resumen(self, n_preview: int = 5) -> str:
        """Pequeño resumen textual para imprimir por consola."""
        return (
            f"Liga: {self._nombre_liga}\n"
            f"Partidos cargados: {len(self._df)}\n"
            f"Rango de fechas: {self._df['date'].min().date()} → {self._df['date'].max().date()}\n"
            f"Equipos distintos: {len(self.listar_equipos())}\n"
            f"Preview:\n{self._df.head(n_preview)}"
        )

def descubrir_ligas(ruta_base_datos: Path) -> list[str]:
    """
    Devuelve una lista de ligas disponibles (carpetas dentro de ruta_base_datos)
    que contengan al menos uno de: temp_ant.csv o temp_act.csv.
    """
    ligas: List[str] = []
    if not ruta_base_datos.exists():
        return ligas
    for sub in ruta_base_datos.iterdir():
        if sub.is_dir():
            if (sub / "temp_ant.csv").exists() or (sub / "temp_act.csv").exists():
                ligas.append(sub.name)
    return sorted(ligas, key=lambda s: s.lower())
#Prueba rapida para paso1
if __name__ == "__main__":
    PRUEBA_RAPIDA = False  # pon True solo cuando quieras probar desde consola
    if PRUEBA_RAPIDA:
        # ... aquí tus prints o pruebas temporales ...
        from pathlib import Path
        from footymodel import ConjuntoPartidos, descubrir_ligas

        ruta_base = Path(r"C:\Estudios\Daniel\App_footymodel\data")

        print("Ligas detectadas:", descubrir_ligas(ruta_base))

        ds_pt = ConjuntoPartidos.desde_liga(ruta_base, "españa")   # o "españa"
        print(ds_pt.resumen())
        print("Algunos equipos:", ds_pt.listar_equipos()[:15])
        pass

#from pathlib import Path
#from footymodel import ConjuntoPartidos, descubrir_ligas

#ruta_base = Path(r"C:\Estudios\Daniel\App_footymodel\data")

#print("Ligas detectadas:", descubrir_ligas(ruta_base))

#ds_pt = ConjuntoPartidos.desde_liga(ruta_base, "españa")   # o "españa"
#print(ds_pt.resumen())
#print("Algunos equipos:", ds_pt.listar_equipos()[:15])

# --- Paso 2: Ponderación por recencia y medias de liga ----------------------
from dataclasses import dataclass
import math
import pandas as pd

class PonderadorRecencia:
    """
    Calcula pesos por recencia:
      w_i = exp( - Δdías_i / τ )
    τ (tau_dias) es el half-life aproximado en días (recomendado: 180).
    """
    def __init__(self, tau_dias: float = 180.0):
        if tau_dias <= 0:
            raise ValueError("tau_dias debe ser > 0")
        self.tau_dias = float(tau_dias)

    def pesos(self, fechas: pd.Series, fecha_referencia: pd.Timestamp | None = None) -> tuple[pd.Series, pd.Timestamp]:
        """
        Devuelve (pesos, fecha_referencia_usada).
        Si no das fecha_referencia, usa la última fecha del dataset (normalizada).
        """
        if fecha_referencia is None:
            fecha_referencia = pd.to_datetime(fechas.max()).normalize()
        # Δdías no negativa
        delta = (fecha_referencia - pd.to_datetime(fechas)).dt.days.clip(lower=0)
        pesos = (-delta / self.tau_dias).map(math.exp)
        return pesos, fecha_referencia

@dataclass
class MediasLiga:
    media_casa: float   # μ_home
    media_fuera: float  # μ_away

def calcular_medias_liga(df_estandar: pd.DataFrame, pesos: pd.Series) -> MediasLiga:
    """
    Calcula μ_home y μ_away ponderadas por 'pesos'.
    df_estandar debe tener columnas: date, home, away, home_goals, away_goals.
    """
    if len(df_estandar) != len(pesos):
        raise ValueError("df_estandar y pesos deben tener la misma longitud")

    suma_pesos = float(pesos.sum())
    if suma_pesos <= 0:
        raise ValueError("La suma de pesos es 0; revisa τ o las fechas de los datos.")

    mu_home = float((pesos * df_estandar["home_goals"]).sum() / suma_pesos)
    mu_away = float((pesos * df_estandar["away_goals"]).sum() / suma_pesos)
    return MediasLiga(media_casa=mu_home, media_fuera=mu_away)

#Prueba rapida para paso2
if __name__ == "__main__":
    PRUEBA_RAPIDA = False  # cambia a True solo para probar rápido
    if PRUEBA_RAPIDA:
        from pathlib import Path

        ruta_base = Path(r"C:\Estudios\Daniel\App_footymodel\data")
        # elige la liga a probar, p. ej. "portugal" o "españa"
        liga = "portugal"

        ds = ConjuntoPartidos.desde_liga(ruta_base, liga)
        print(ds.resumen(3))

        ponderador = PonderadorRecencia(tau_dias=180)
        pesos, fecha_ref = ponderador.pesos(ds.df["date"])
        medias = calcular_medias_liga(ds.df, pesos)

        print(f"\nFecha de referencia: {fecha_ref.date()}")
        print(f"μ_casa (liga {liga}):  {medias.media_casa:.3f}")
        print(f"μ_fuera (liga {liga}): {medias.media_fuera:.3f}")

# --- Paso 3: Tasas por equipo (shrinkage) y factores relativos ---------------
from dataclasses import dataclass
import pandas as pd

@dataclass
class TasasEquipos:
    """
    Tasas por equipo (goles/partido) con shrinkage hacia la media de liga.
    Índices: nombre de equipo (str)
    """
    r_casa_ataque: pd.Series   # goles que ANOTA en casa
    r_casa_defensa: pd.Series  # goles que RECIBE en casa (anotados por el visitante)
    r_fuera_ataque: pd.Series  # goles que ANOTA fuera
    r_fuera_defensa: pd.Series # goles que RECIBE fuera (anotados por el local)

class CalculadoraTasasEquipos:
    """
    Calcula tasas por equipo aplicando:
      r = ( Σ w * goles  + α * μ ) / ( Σ w + α )
    donde μ es la media de liga adecuada al rol:
      - μ_casa para ataque en casa y defensa del rival FUERA
      - μ_fuera para ataque fuera y defensa del rival EN CASA
    """
    def __init__(self, alpha: float = 8.0, piso_tasa: float = 0.05):
        if alpha < 0:
            raise ValueError("alpha debe ser >= 0")
        self.alpha = float(alpha)
        self.piso_tasa = float(piso_tasa)

    def calcular_tasas(self, df_estandar: pd.DataFrame, pesos: pd.Series, medias_liga: MediasLiga) -> TasasEquipos:
        if len(df_estandar) != len(pesos):
            raise ValueError("df_estandar y pesos deben tener la misma longitud")

        # Preparamos columnas ponderadas para agregar sin lambdas complejas
        tmp = df_estandar.copy()
        tmp = tmp.assign(peso=pesos.values)
        tmp["w_home_goals"] = tmp["peso"] * tmp["home_goals"]
        tmp["w_away_goals"] = tmp["peso"] * tmp["away_goals"]

        # Agregaciones por equipo en rol CASA
        g_casa = tmp.groupby("home").agg(
            suma_pesos=("peso", "sum"),
            goles_ponderados_marcados=("w_home_goals", "sum"),
            goles_ponderados_recibidos=("w_away_goals", "sum"),
        )

        # Agregaciones por equipo en rol FUERA
        g_fuera = tmp.groupby("away").agg(
            suma_pesos=("peso", "sum"),
            goles_ponderados_marcados=("w_away_goals", "sum"),
            goles_ponderados_recibidos=("w_home_goals", "sum"),
        )

        α = self.alpha
        μc = medias_liga.media_casa
        μf = medias_liga.media_fuera
        eps = 1e-12

        # Tasas con shrinkage
        r_casa_ataque = (g_casa["goles_ponderados_marcados"] + α * μc) / (g_casa["suma_pesos"] + α + eps)
        r_casa_defensa = (g_casa["goles_ponderados_recibidos"] + α * μf) / (g_casa["suma_pesos"] + α + eps)

        r_fuera_ataque = (g_fuera["goles_ponderados_marcados"] + α * μf) / (g_fuera["suma_pesos"] + α + eps)
        r_fuera_defensa = (g_fuera["goles_ponderados_recibidos"] + α * μc) / (g_fuera["suma_pesos"] + α + eps)

        # Pisos (evitar ceros duros)
        r_casa_ataque = r_casa_ataque.clip(lower=self.piso_tasa)
        r_casa_defensa = r_casa_defensa.clip(lower=self.piso_tasa)
        r_fuera_ataque = r_fuera_ataque.clip(lower=self.piso_tasa)
        r_fuera_defensa = r_fuera_defensa.clip(lower=self.piso_tasa)

        return TasasEquipos(
            r_casa_ataque=r_casa_ataque,
            r_casa_defensa=r_casa_defensa,
            r_fuera_ataque=r_fuera_ataque,
            r_fuera_defensa=r_fuera_defensa,
        )

@dataclass
class FactoresEquipo:
    """
    Factores relativos respecto a la media de liga (1.0 = promedio liga):
      - a_casa = r_casa_ataque / μ_casa
      - d_casa = r_casa_defensa / μ_fuera
      - a_fuera = r_fuera_ataque / μ_fuera
      - d_fuera = r_fuera_defensa / μ_casa
    """
    a_casa: pd.Series
    d_casa: pd.Series
    a_fuera: pd.Series
    d_fuera: pd.Series

def calcular_factores_equipo(tasas: TasasEquipos, medias_liga: MediasLiga) -> FactoresEquipo:
    μc = medias_liga.media_casa
    μf = medias_liga.media_fuera
    eps = 1e-12

    a_casa  = tasas.r_casa_ataque  / (μc + eps)
    d_casa  = tasas.r_casa_defensa / (μf + eps)
    a_fuera = tasas.r_fuera_ataque / (μf + eps)
    d_fuera = tasas.r_fuera_defensa/ (μc + eps)

    return FactoresEquipo(a_casa=a_casa, d_casa=d_casa, a_fuera=a_fuera, d_fuera=d_fuera)

#prueba rapida paso3
if __name__ == "__main__":
    PRUEBA_RAPIDA = False
    PRUEBA_P3 = False  # cambia a True para probar Paso 3

    if PRUEBA_P3:
        from pathlib import Path

        ruta_base = Path(r"C:\Estudios\Daniel\App_footymodel\data")
        liga = "portugal"

        ds = ConjuntoPartidos.desde_liga(ruta_base, liga)

        ponderador = PonderadorRecencia(tau_dias=180)
        pesos, fecha_ref = ponderador.pesos(ds.df["date"])
        medias = calcular_medias_liga(ds.df, pesos)

        calc_tasas = CalculadoraTasasEquipos(alpha=8.0, piso_tasa=0.05)
        tasas = calc_tasas.calcular_tasas(ds.df, pesos, medias)

        factores = calcular_factores_equipo(tasas, medias)

        print(f"Fecha de referencia: {fecha_ref.date()}")
        print(f"μ_casa={medias.media_casa:.3f}  μ_fuera={medias.media_fuera:.3f}")
        equipos = ds.listar_equipos()[:5]  # muestra 5 equipos ejemplo

        print("\nEjemplo de tasas (goles/partido) con shrinkage:")
        for e in equipos:
            print(f"  {e:15s}  r_casa_att={tasas.r_casa_ataque.get(e, float('nan')):.3f}  "
                  f"r_casa_def={tasas.r_casa_defensa.get(e, float('nan')):.3f}  "
                  f"r_fuera_att={tasas.r_fuera_ataque.get(e, float('nan')):.3f}  "
                  f"r_fuera_def={tasas.r_fuera_defensa.get(e, float('nan')):.3f}")

        print("\nEjemplo de factores relativos (1.0 = media liga):")
        for e in equipos:
            print(f"  {e:15s}  a_casa={factores.a_casa.get(e, float('nan')):.3f}  "
                  f"d_casa={factores.d_casa.get(e, float('nan')):.3f}  "
                  f"a_fuera={factores.a_fuera.get(e, float('nan')):.3f}  "
                  f"d_fuera={factores.d_fuera.get(e, float('nan')):.3f}")


# --- Paso 4: Poisson + Predictor y tabla 0–4+ -------------------------------
from dataclasses import dataclass
import math
import pandas as pd
from difflib import get_close_matches

class HerramientasPoisson:
    @staticmethod
    def pmf_poisson(k: int, lam: float) -> float:
        if lam < 0:
            return 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    @staticmethod
    def vector_pmf(lam: float, max_k: int) -> list[float]:
        return [HerramientasPoisson.pmf_poisson(k, lam) for k in range(max_k + 1)]

    @staticmethod
    def tabla_0a4_mas(lam: float) -> pd.Series:
        """
        Devuelve una serie con P(0), P(1), P(2), P(3), P(4+)
        (Poisson independiente con media 'lam').
        """
        p0 = HerramientasPoisson.pmf_poisson(0, lam)
        p1 = HerramientasPoisson.pmf_poisson(1, lam)
        p2 = HerramientasPoisson.pmf_poisson(2, lam)
        p3 = HerramientasPoisson.pmf_poisson(3, lam)
        p_4mas = 1.0 - (p0 + p1 + p2 + p3)
        return pd.Series(
            [p0, p1, p2, p3, p_4mas],
            index=["0", "1", "2", "3", "4+"],
            dtype="float64",
        )

@dataclass
class ResultadoPrediccion:
    equipo_local: str
    equipo_visitante: str
    lambda_local: float
    lambda_visitante: float
    prob_local_marca: float
    prob_visitante_marca: float
    prob_btts: float
    esperado_goles_totales: float
    tabla_goles_local: pd.Series   # índice: ["0","1","2","3","4+"]
    tabla_goles_visitante: pd.Series

class Predictor:
    """
    Calcula intensidades y probabilidades para A (local) vs B (visitante) usando:
      λ_local = μ_casa  * a_casa[A] * d_fuera[B]
      λ_visit = μ_fuera * a_fuera[B] * d_casa[A]
    """
    def __init__(self, factores: FactoresEquipo, medias_liga: MediasLiga):
        self.factores = factores
        self.medias = medias_liga

    @staticmethod
    def _validar_equipo(nombre: str, indice: pd.Index, rol: str) -> None:
        if nombre not in indice:
            suger = get_close_matches(nombre, indice.tolist(), n=5, cutoff=0.6)
            raise ValueError(
                f"Equipo '{nombre}' no encontrado para rol {rol}. "
                f"Sugerencias: {suger}"
            )

    def predecir(self, equipo_local: str, equipo_visitante: str) -> ResultadoPrediccion:
        # Validaciones y sugerencias
        self._validar_equipo(equipo_local,   self.factores.a_casa.index,  "local")
        self._validar_equipo(equipo_visitante, self.factores.a_fuera.index, "visitante")

        # Intensidades esperadas
        lam_local = float(
            self.medias.media_casa
            * self.factores.a_casa.loc[equipo_local]
            * self.factores.d_fuera.loc[equipo_visitante]
        )
        lam_visit = float(
            self.medias.media_fuera
            * self.factores.a_fuera.loc[equipo_visitante]
            * self.factores.d_casa.loc[equipo_local]
        )

        # Probabilidades clave (independencia)
        p_local_marca = 1.0 - math.exp(-lam_local)
        p_visit_marca = 1.0 - math.exp(-lam_visit)
        p_btts = p_local_marca * p_visit_marca
        esperado_totales = lam_local + lam_visit

        # Tablas 0–4+ por equipo
        tabla_local = HerramientasPoisson.tabla_0a4_mas(lam_local)
        tabla_visit = HerramientasPoisson.tabla_0a4_mas(lam_visit)

        return ResultadoPrediccion(
            equipo_local=equipo_local,
            equipo_visitante=equipo_visitante,
            lambda_local=lam_local,
            lambda_visitante=lam_visit,
            prob_local_marca=p_local_marca,
            prob_visitante_marca=p_visit_marca,
            prob_btts=p_btts,
            esperado_goles_totales=esperado_totales,
            tabla_goles_local=tabla_local,
            tabla_goles_visitante=tabla_visit,
        )

def mostrar_resumen_prediccion(res: ResultadoPrediccion) -> None:
    # Etiquetas para P(marca) y P(no marca)
    pL, pV = res.prob_local_marca, res.prob_visitante_marca
    pL_no, pV_no = (1.0 - pL), (1.0 - pV)
    etq_L = clasificar_prob_marca(pL)
    etq_V = clasificar_prob_marca(pV)
    etq_L_no = clasificar_prob_no_marca(pL_no)
    etq_V_no = clasificar_prob_no_marca(pV_no)

    print("\n=== Resumen de predicción ===")
    print(f"Partido: {res.equipo_local} (local) vs {res.equipo_visitante} (visitante)")
    print(f"λ_{res.equipo_local} (local):     {res.lambda_local:.3f}")
    print(f"λ_{res.equipo_visitante} (visit): {res.lambda_visitante:.3f}")

    print(f"P({res.equipo_local} marca):     {pL:.1%}  → {etq_L}  (NO marca: {pL_no:.1%}, {etq_L_no})")
    print(f"P({res.equipo_visitante} marca): {pV:.1%}  → {etq_V}  (NO marca: {pV_no:.1%}, {etq_V_no})")
    print(f"P(BTTS):                         {res.prob_btts:.1%}")
    print(f"Esperado goles totales:          {res.esperado_goles_totales:.3f}")

    # Tabla comparativa 0–4+ por equipo (sin applymap deprecado)
    tabla = pd.DataFrame(
        {
            res.equipo_local: res.tabla_goles_local,
            res.equipo_visitante: res.tabla_goles_visitante,
        }
    )
    tabla_fmt = tabla.map(lambda x: f"{x:.1%}")
    print("\nProbabilidades de goles por equipo (0–4+):")
    print(tabla_fmt.to_string())


# --- Paso 4(mejorado): Etiquetas y últimos resultados ---------------------------------
import pandas as pd

def clasificar_prob_marca(p: float) -> str:
    """Etiqueta para P(marca)."""
    if p >= 0.80:
        return "Muy probable que marque"
    elif p >= 0.65:
        return "Probable que marque"
    elif p >= 0.50:
        return "Equilibrado (leve sí)"
    elif p >= 0.35:
        return "Equilibrado (leve no)"
    elif p >= 0.20:
        return "Poco probable que marque"
    else:
        return "Muy improbable que marque"

def clasificar_prob_no_marca(p_no: float) -> str:
    """Etiqueta para P(NO marca)."""
    if p_no >= 0.80:
        return "Muy probable que NO marque"
    elif p_no >= 0.65:
        return "Probable que NO marque"
    elif p_no >= 0.50:
        return "Equilibrado (leve no)"
    elif p_no >= 0.35:
        return "Equilibrado (leve sí)"
    elif p_no >= 0.20:
        return "Poco probable que NO marque"
    else:
        return "Muy improbable que NO marque"

def ultimos_resultados_equipo(df_estandar: pd.DataFrame, equipo: str, n: int = 5) -> pd.DataFrame:
    """
    Devuelve los últimos n partidos del equipo (más recientes primero),
    con columnas: fecha, condición, rival, gf, gc, resultado (G/E/P).
    """
    sub = df_estandar[(df_estandar["home"] == equipo) | (df_estandar["away"] == equipo)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["fecha", "condición", "rival", "gf", "gc", "resultado"])

    sub = sub.sort_values("date", ascending=False).head(n).copy()

    def fila_a_res(fila) -> dict:
        es_local = (fila["home"] == equipo)
        condicion = "Local" if es_local else "Visitante"
        rival = fila["away"] if es_local else fila["home"]
        gf = int(fila["home_goals"]) if es_local else int(fila["away_goals"])
        gc = int(fila["away_goals"]) if es_local else int(fila["home_goals"])
        if gf > gc:
            res = "G"
        elif gf == gc:
            res = "E"
        else:
            res = "P"
        return {
            "fecha": fila["date"].date(),
            "condición": condicion,
            "rival": rival,
            "gf": gf,
            "gc": gc,
            "resultado": res,
        }

    filas = [fila_a_res(row) for _, row in sub.iterrows()]
    return pd.DataFrame(filas)

def imprimir_ultimos_resultados(df_estandar: pd.DataFrame, equipo: str, n: int = 5) -> None:
    """Imprime una tablita con los últimos n partidos del equipo."""
    hist = ultimos_resultados_equipo(df_estandar, equipo, n=n)
    print(f"\nÚltimos {len(hist)} resultados de {equipo} (más recientes primero):")
    if hist.empty:
        print("  (sin partidos)")
    else:
        print(hist.to_string(index=False))
        
def ultimos_resultados_por_condicion(df_estandar: pd.DataFrame, equipo: str, condicion: str, n: int = 5) -> pd.DataFrame:
    """
    Devuelve los últimos n partidos del equipo *en la condición indicada* ('Local' o 'Visitante'),
    ordenados del más reciente al más antiguo.
    Columnas: fecha, condición, rival, gf, gc, resultado (G/E/P).
    """
    cond = condicion.strip().lower()
    if cond not in ("local", "visitante"):
        raise ValueError("condicion debe ser 'Local' o 'Visitante'")

    if cond == "local":
        sub = df_estandar[df_estandar["home"] == equipo].copy()
        etiqueta = "Local"
        def fila_a_res(fila):
            gf = int(fila["home_goals"]); gc = int(fila["away_goals"])
            rival = fila["away"]
            res = "G" if gf > gc else ("E" if gf == gc else "P")
            return {"fecha": fila["date"].date(), "condición": etiqueta, "rival": rival, "gf": gf, "gc": gc, "resultado": res}
    else:
        sub = df_estandar[df_estandar["away"] == equipo].copy()
        etiqueta = "Visitante"
        def fila_a_res(fila):
            gf = int(fila["away_goals"]); gc = int(fila["home_goals"])
            rival = fila["home"]
            res = "G" if gf > gc else ("E" if gf == gc else "P")
            return {"fecha": fila["date"].date(), "condición": etiqueta, "rival": rival, "gf": gf, "gc": gc, "resultado": res}

    if sub.empty:
        return pd.DataFrame(columns=["fecha", "condición", "rival", "gf", "gc", "resultado"])

    sub = sub.sort_values("date", ascending=False).head(n).copy()
    filas = [fila_a_res(row) for _, row in sub.iterrows()]
    return pd.DataFrame(filas)

def imprimir_ultimos_resultados_por_condicion(df_estandar: pd.DataFrame, equipo: str, condicion: str, n: int = 5) -> None:
    """Imprime los últimos n resultados del equipo *en la condición indicada*."""
    hist = ultimos_resultados_por_condicion(df_estandar, equipo, condicion, n=n)
    print(f"\nÚltimos {len(hist)} resultados de {equipo} como {condicion} (más recientes primero):")
    if hist.empty:
        print("  (sin partidos en esa condición)")
    else:
        print(hist.to_string(index=False))


def imprimir_factores_partido(factores: FactoresEquipo, medias: MediasLiga,
                              equipo_local: str, equipo_visitante: str) -> None:
    """
    Imprime los factores relativos (1.0 = media liga) que intervienen en las λ:
      - Local:  a_casa(local) y d_casa(local)
      - Visit:  a_fuera(visit) y d_fuera(visit)
    y muestra la descomposición de λ_local y λ_visitante.
    """
    # Factores del local (en casa)
    aL = float(factores.a_casa.loc[equipo_local])
    dL = float(factores.d_casa.loc[equipo_local])

    # Factores del visitante (fuera)
    aV = float(factores.a_fuera.loc[equipo_visitante])
    dV = float(factores.d_fuera.loc[equipo_visitante])

    mu_c = medias.media_casa
    mu_f = medias.media_fuera

    lam_local_calc = mu_c * aL * dV
    lam_visit_calc = mu_f * aV * dL

    print("\nFactores relativos empleados (1.0 = media de liga):")
    print(f"  {equipo_local} (LOCAL)   → ataque a_casa={aL:.3f} | defensa d_casa={dL:.3f}  (d<1 ⇒ mejor defensa)")
    print(f"  {equipo_visitante} (VISIT) → ataque a_fuera={aV:.3f} | defensa d_fuera={dV:.3f}  (d<1 ⇒ mejor defensa)")

    print("\nDescomposición de intensidades (λ):")
    print(f"  λ_{equipo_local} = μ_casa({mu_c:.3f}) × a_casa({aL:.3f}) × d_fuera({dV:.3f}) = {lam_local_calc:.3f}")
    print(f"  λ_{equipo_visitante} = μ_fuera({mu_f:.3f}) × a_fuera({aV:.3f}) × d_casa({dL:.3f}) = {lam_visit_calc:.3f}")


#prueba rapida paso4
if __name__ == "__main__":
    PRUEBA_RAPIDA = False
    PRUEBA_P3 = False
    PRUEBA_P4 = False   # ← pon True para probar el Paso 4

    if PRUEBA_P4:
        from pathlib import Path

        ruta_base = Path(r"C:\Estudios\Daniel\App_footymodel\data")
        liga = "portugal"

        ds = ConjuntoPartidos.desde_liga(ruta_base, liga)
        ponderador = PonderadorRecencia(tau_dias=180)
        pesos, fecha_ref = ponderador.pesos(ds.df["date"])
        medias = calcular_medias_liga(ds.df, pesos)

        calc_tasas = CalculadoraTasasEquipos(alpha=8.0, piso_tasa=0.05)
        tasas = calc_tasas.calcular_tasas(ds.df, pesos, medias)
        factores = calcular_factores_equipo(tasas, medias)

        predictor = Predictor(factores, medias)

        # Cambia por dos equipos que existan en ds.listar_equipos()
        equipo_local = "Benfica"
        equipo_visitante = "Porto"

        res = predictor.predecir(equipo_local, equipo_visitante)
        mostrar_resumen_prediccion(res)

# --- Paso 5: Interfaz interactiva por consola --------------------------------
from difflib import get_close_matches
from pathlib import Path

class InterfazInteractiva:
    def __init__(self, ruta_base_datos: Path):
        self.ruta_base = ruta_base_datos

    def elegir_liga(self) -> str:
        ligas = descubrir_ligas(self.ruta_base)
        if not ligas:
            raise FileNotFoundError(f"No se encontraron ligas en {self.ruta_base}")
        print("\nSeleccione la liga:")
        for i, liga in enumerate(ligas, start=1):
            print(f"  {i}. {liga}")
        while True:
            s = input("Número de liga: ").strip()
            if s.isdigit():
                idx = int(s)
                if 1 <= idx <= len(ligas):
                    return ligas[idx - 1]
            print("Entrada inválida. Intente de nuevo.")

    def pedir_equipo(self, ds: ConjuntoPartidos, rol: str) -> str:
        equipos = ds.listar_equipos()
        set_equipos = set(equipos)
        while True:
            nombre = input(f"Ingrese {rol} (nombre exacto, '?' para listar): ").strip()
            if nombre == "?":
                print(", ".join(equipos))
                continue
            if nombre in set_equipos:
                return nombre
            suger = get_close_matches(nombre, equipos, n=5, cutoff=0.6)
            print(f"Equipo no encontrado. Sugerencias: {suger}")

    def correr(self, tau: float = 180.0, alpha: float = 8.0):
        liga = self.elegir_liga()
        ds = ConjuntoPartidos.desde_liga(self.ruta_base, liga)
        ponderador = PonderadorRecencia(tau_dias=tau)
        pesos, fecha_ref = ponderador.pesos(ds.df["date"])
        medias = calcular_medias_liga(ds.df, pesos)
        calc_tasas = CalculadoraTasasEquipos(alpha=alpha, piso_tasa=0.05)
        tasas = calc_tasas.calcular_tasas(ds.df, pesos, medias)
        factores = calcular_factores_equipo(tasas, medias)
        predictor = Predictor(factores, medias)

        print(f"\nLiga '{liga}' cargada.")
        print(f"Fecha de referencia: {fecha_ref.date()}  |  μ_casa={medias.media_casa:.3f}  μ_fuera={medias.media_fuera:.3f}")

        local = self.pedir_equipo(ds, "EQUIPO LOCAL")
        visitante = self.pedir_equipo(ds, "EQUIPO VISITANTE")

        resultado = predictor.predecir(local, visitante)
        mostrar_resumen_prediccion(resultado)
                # Mostrar últimos 5 resultados de ambos equipos
        imprimir_ultimos_resultados(ds.df, local, n=5)
        imprimir_ultimos_resultados(ds.df, visitante, n=5)
        
                # Últimos 5 según la condición del partido:
        imprimir_ultimos_resultados_por_condicion(ds.df, local, "Local", n=5)
        imprimir_ultimos_resultados_por_condicion(ds.df, visitante, "Visitante", n=5)
        
        # Mostrar factores relativos usados y descomposición de λ
        imprimir_factores_partido(factores, medias, local, visitante)



###### EJECUTADOR INTERACTIVO #####
if __name__ == "__main__":
    INTERACTIVO = True  # pon True cuando quieras usar el menú
    if INTERACTIVO:
        base = Path(r"C:\Estudios\Daniel\App_footymodel\data")  # tu ruta base
        app = InterfazInteractiva(base)
        app.correr(tau=180.0, alpha=8.0)
