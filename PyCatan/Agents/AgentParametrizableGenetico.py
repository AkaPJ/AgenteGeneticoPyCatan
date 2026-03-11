"""
AgentParametrizableGenetico.py
==============================
Agente que usa un Algoritmo Genético (GA) para optimizar los parámetros
de AgentParametrizable y maximizar el winrate contra los oponentes estándar.

Uso básico::

    from Agents.AgentParametrizableGenetico import AgentParametrizableGenetico

    agente = AgentParametrizableGenetico(agent_id=0)
    agente.train(n_generaciones=50, verbose=True)
    agente.guardar_params("mejor_params.json")

    # El agente ya tiene los mejores parámetros aplicados y puede usarse
    # directamente en partidas. También se puede recargar desde fichero:
    agente2 = AgentParametrizableGenetico.desde_fichero("mejor_params.json", agent_id=0)

Características del GA
-----------------------
- Pool de oponentes fijo por generación (CRN – Common Random Numbers)
- BLX-α crossover
- Mutación gaussiana adaptativa con sigma individual por gen
- Elitismo
- Selección por torneo
- Reinicio parcial de población por estancamiento
- Annealing de sigma
- Fitness con componente de ranking penalizado
- Rotación cíclica de todos los oponentes para evaluación más representativa
- Evaluación final con todos los oponentes disponibles
"""

import json
import os
import random
import sys
import concurrent.futures
from typing import Dict, List, Optional, Tuple

from Agents.AgentParametrizable import AgentParametrizable
from Agents.AdrianHerasAgent import AdrianHerasAgent
from Agents.AlexPastorAgent import AlexPastorAgent
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent
from Agents.CarlesZaidaAgent import CarlesZaidaAgent
from Agents.CrabisaAgent import CrabisaAgent
from Agents.EdoAgent import EdoAgent
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent
from Agents.SigmaAgent import SigmaAgent
from Agents.TristanAgent import TristanAgent
from Managers.GameDirector import GameDirector

# ---------------------------------------------------------------------------
# Pool de agentes oponentes disponibles
# ---------------------------------------------------------------------------
_OPPONENT_AGENTS = [
    AdrianHerasAgent,
    AlexPastorAgent,
    AlexPelochoJaimeAgent,
    CarlesZaidaAgent,
    CrabisaAgent,
    EdoAgent,
    PabloAleixAlexAgent,
    SigmaAgent,
    TristanAgent,
]

# ---------------------------------------------------------------------------
# Espacio de genes: nombre -> (default, min, max, tipo)
# ---------------------------------------------------------------------------
_GENE_SPACE: Dict[str, tuple] = {
    # Pesos de puntuación de nodos: se usan como multiplicadores en suma
    # ponderada, no como probabilidades → se permite hasta 2.0 para dar
    # más margen a estrategias que priorizan fuertemente una dimensión.
    "w_production":               (0.5,  0.0, 2.0, float),
    "w_diversity":                (0.3,  0.0, 2.0, float),
    "w_port_access":              (0.2,  0.0, 2.0, float),
    # Multiplicadores de puntuación de construcción: score = base * umbral,
    # extender a 2.0 permite estrategias muy agresivas de ciudad/carretera.
    "city_upgrade_threshold":     (0.6,  0.0, 2.0, float),
    "road_expansion_weight":      (0.4,  0.0, 2.0, float),
    # Factor de mezcla [0,1]: 1.0 = colocar el ladrón en el hex más productivo
    # del rival; 0.0 = evitar los propios hexes. Valores >1 distorsionarían la lógica.
    "robber_aggression":          (0.7,  0.0, 1.0, float),
    # Umbral de ratio: valores >1 harían que nunca se acepten tratos
    "trade_acceptance_threshold": (0.55, 0.0, 1.0, float),
    # Probabilidad [0,1]
    "dev_card_eagerness":         (0.5,  0.0, 1.0, float),
    # Discreto
    "discard_priority":           (1,    0,   2,   int),
    # Factor de mezcla [0,1]: 1.0 = priorizar expansion; 0.0 = priorizar ciudades
    "expansion_vs_consolidation": (0.5,  0.0, 1.0, float),
    # Enteros con rango ampliado
    "min_road_before_settle":     (2,    0,   6,   int),
    "vp_urgency_threshold":       (8,    5,   10,  int),
}

_GENE_NAMES: List[str] = list(_GENE_SPACE.keys())

# Cota superior para semillas aleatorias (2^31 = límite seguro para Random.seed)
_MAX_SEED = 2 ** 31


# ---------------------------------------------------------------------------
# Funciones de módulo (picklables para multiprocessing)
# ---------------------------------------------------------------------------

def _genoma_aleatorio(rng: random.Random) -> dict:
    """Genera un genoma aleatorio respetando los rangos de _GENE_SPACE."""
    genoma = {}
    for nombre, (default, lo, hi, tipo) in _GENE_SPACE.items():
        if tipo is int:
            genoma[nombre] = rng.randint(lo, hi)
        else:
            genoma[nombre] = rng.uniform(lo, hi)
    return genoma


def _simular_partida(
    genoma: dict,
    opponents: list,
    position: int,
    seed: int,
) -> Tuple[int, float, int]:
    """
    Simula una partida con el agente parametrizado por ``genoma`` en la
    posición ``position`` contra los oponentes dados.

    Devuelve ``(victoria: 0|1, puntos: float, rank: 1-4)``.
    """
    try:
        random.seed(seed)

        class _AgenteEval(AgentParametrizable):
            def __init__(self, agent_id):
                super().__init__(agent_id, **genoma)

        agents = list(opponents)
        agents.insert(position, _AgenteEval)

        gd = GameDirector(agents=agents, max_rounds=200, store_trace=False)
        trace = gd.game_start(print_outcome=False)

        last_round = max(
            trace["game"].keys(), key=lambda r: int(r.split("_")[-1])
        )
        last_turn = max(
            trace["game"][last_round].keys(),
            key=lambda t: int(t.split("_")[-1].lstrip("P")),
        )
        vps = trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

        agent_id_str = f"J{position}"
        points = int(vps.get(agent_id_str, 0))
        winner = max(vps, key=lambda p: int(vps[p]))
        victory = 1 if winner == agent_id_str else 0

        ordenados = sorted(vps.items(), key=lambda x: int(x[1]), reverse=True)
        rank = 4
        for idx, (jugador, _) in enumerate(ordenados, start=1):
            if jugador == agent_id_str:
                rank = idx
                break

        return (victory, float(points), rank)

    except Exception as e:
        print(f"[GA] _simular_partida error: {repr(e)}", file=sys.stderr)
        return (0, 0.0, 4)


def _evaluar_individuo_paralelo(args: tuple) -> Tuple[int, float, int, int]:
    """
    Wrapper picklable para evaluar un único individuo del GA.

    Ejecuta ``k_partidas`` partidas rotando oponentes cíclicamente si el pool
    tiene más de 3 agentes, y devuelve los acumulados
    ``(victorias, puntos, rank_sum, n_partidas)``.
    """
    genoma, opponents_for_gen, k_partidas, seed_base, opponents_mode = args
    rng = random.Random(seed_base)
    victorias = 0
    puntos_total = 0.0
    rank_total = 0
    pool = opponents_for_gen
    pool_size = len(pool)

    for k in range(k_partidas):
        position = k % 4

        # Rotación cíclica del pool cuando es grande (mejora representatividad)
        if opponents_mode == "pool_fijo" and pool_size > 3:
            start = (k * 3) % pool_size
            opp = [pool[(start + j) % pool_size] for j in range(3)]
        elif pool_size >= 3:
            opp = pool[:3]
        else:
            # Pool pequeño: repetir para completar 3 oponentes
            opp = (pool * 4)[:3]

        game_seed = rng.randint(0, _MAX_SEED)
        v, pts, rank = _simular_partida(genoma, opp, position, game_seed)
        victorias += v
        puntos_total += pts
        rank_total += rank

    return (victorias, puntos_total, rank_total, k_partidas)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class AgentParametrizableGenetico(AgentParametrizable):
    """
    Agente de Catan que optimiza sus parámetros mediante un Algoritmo Genético.

    Hereda de :class:`AgentParametrizable`, por lo que puede usarse
    directamente en partidas una vez entrenado (o tras cargar parámetros desde
    fichero con :meth:`desde_fichero`).

    El GA maximiza el winrate contra el conjunto de agentes estándar del
    proyecto usando:

    - Pool de oponentes fijo por generación (CRN)
    - BLX-α crossover
    - Mutación gaussiana con sigma adaptativo por gen
    - Elitismo
    - Selección por torneo
    - Reinicio parcial de población por estancamiento
    - Annealing de sigma
    - Fitness con componente de ranking penalizado
    """

    def __init__(self, agent_id: int = 0, **params):
        super().__init__(agent_id, **params)
        self._mejor_genoma: Optional[dict] = None
        self._mejor_fitness: float = -float("inf")

    # =========================================================================
    # Persistencia
    # =========================================================================

    def guardar_params(self, fichero: str) -> None:
        """Guarda los mejores parámetros encontrados en un fichero JSON."""
        params = self._mejor_genoma if self._mejor_genoma is not None else self.get_params()
        with open(fichero, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"[AgentParametrizableGenetico] Parámetros guardados en '{fichero}'")

    @classmethod
    def desde_fichero(
        cls, fichero: str, agent_id: int = 0
    ) -> "AgentParametrizableGenetico":
        """Crea un agente cargando sus parámetros desde un fichero JSON."""
        with open(fichero, "r", encoding="utf-8") as f:
            params = json.load(f)
        # Asegurar tipos enteros donde corresponda
        for nombre in _GENE_NAMES:
            if nombre in params and _GENE_SPACE[nombre][3] is int:
                params[nombre] = int(params[nombre])
        agente = cls(agent_id, **params)
        agente._mejor_genoma = dict(params)
        return agente

    def get_params(self) -> dict:
        """Devuelve los parámetros actuales del agente como diccionario."""
        return {nombre: getattr(self, nombre) for nombre in _GENE_NAMES}

    def __repr__(self) -> str:
        params = self.get_params()
        return (
            f"AgentParametrizableGenetico(id={self.id}, "
            f"fitness={self._mejor_fitness:.4f}, params={params})"
        )

    # =========================================================================
    # Operadores del Algoritmo Genético
    # =========================================================================

    @staticmethod
    def _blx_alpha_crossover(
        g1: dict, g2: dict, alpha: float, rng: random.Random
    ) -> Tuple[dict, dict]:
        """
        BLX-α crossover: genera dos hijos interpolando entre los padres
        con un margen de extensión ``alpha`` más allá del intervalo [g1, g2].
        """
        h1: dict = {}
        h2: dict = {}
        for nombre in _GENE_NAMES:
            _, lo, hi, tipo = _GENE_SPACE[nombre]
            v1, v2 = g1[nombre], g2[nombre]
            d = abs(v2 - v1)
            blo = min(v1, v2) - alpha * d
            bhi = max(v1, v2) + alpha * d
            eff_lo = max(lo, blo)
            eff_hi = min(hi, bhi)
            # Si el intervalo efectivo es degenerado, usar el rango completo
            if eff_lo > eff_hi:
                eff_lo, eff_hi = float(lo), float(hi)
            c1 = rng.uniform(eff_lo, eff_hi)
            c2 = rng.uniform(eff_lo, eff_hi)
            if tipo is int:
                h1[nombre] = int(round(max(lo, min(hi, c1))))
                h2[nombre] = int(round(max(lo, min(hi, c2))))
            else:
                h1[nombre] = max(lo, min(hi, c1))
                h2[nombre] = max(lo, min(hi, c2))
        return h1, h2

    @staticmethod
    def _mutar_adaptativo(
        genoma: dict,
        sigmas: dict,
        prob_mutacion: float,
        rng: random.Random,
    ) -> dict:
        """
        Mutación gaussiana adaptativa: cada gen muta con probabilidad
        ``prob_mutacion`` usando su sigma individual, escalado por el rango
        del gen.  Esto permite que genes sensibles evolucionen más despacio
        y genes insensibles más rápido.
        """
        nuevo = dict(genoma)
        for nombre in _GENE_NAMES:
            if rng.random() < prob_mutacion:
                _, lo, hi, tipo = _GENE_SPACE[nombre]
                rango = (hi - lo) if hi != lo else 1.0
                ruido = rng.gauss(0.0, sigmas[nombre] * rango)
                valor = genoma[nombre] + ruido
                valor = max(lo, min(hi, valor))
                if tipo is int:
                    nuevo[nombre] = int(round(valor))
                else:
                    nuevo[nombre] = valor
        return nuevo

    @staticmethod
    def _torneo(
        poblacion: list,
        fitnesses: list,
        k: int,
        rng: random.Random,
    ) -> dict:
        """Selección por torneo de tamaño ``k``: devuelve el mejor individuo."""
        candidatos = rng.sample(range(len(poblacion)), min(k, len(poblacion)))
        mejor = max(candidatos, key=lambda i: fitnesses[i])
        return dict(poblacion[mejor])

    @staticmethod
    def _reinicio_parcial(
        poblacion: list,
        fitnesses: list,
        fraccion: float,
        rng: random.Random,
    ) -> list:
        """
        Mantiene el top ``(1 - fraccion)`` de la población y reemplaza el
        resto con individuos aleatorios.  Permite escapar de óptimos locales
        sin descartar los mejores individuos encontrados hasta el momento.
        """
        n = len(poblacion)
        n_conservar = max(1, int(n * (1.0 - fraccion)))
        ordenados = sorted(range(n), key=lambda i: fitnesses[i], reverse=True)
        nueva_pob = [dict(poblacion[i]) for i in ordenados[:n_conservar]]
        while len(nueva_pob) < n:
            nueva_pob.append(_genoma_aleatorio(rng))
        return nueva_pob

    @staticmethod
    def _calcular_fitness_winrate_first(
        victorias: int,
        puntos_total: float,
        rank_total: int,
        n_partidas: int,
    ) -> float:
        """
        Fitness centrado en winrate con bonus por puntuación media y
        penalización suave por ranking promedio::

            fitness = win_rate
                    + 0.02 * min(avg_pts / 10, 1)
                    - 0.01 * (avg_rank - 1) / 3

        Esto desempata entre individuos con igual winrate priorizando los que
        quedan más arriba en el ranking (más puntos aunque no ganen).
        """
        if n_partidas == 0:
            return 0.0
        win_rate = victorias / n_partidas
        avg_pts = puntos_total / n_partidas
        avg_rank = rank_total / n_partidas
        return win_rate + 0.02 * min(avg_pts / 10.0, 1.0) - 0.01 * (avg_rank - 1) / 3.0

    def _evaluar_poblacion(
        self,
        poblacion: list,
        opponents_for_gen: list,
        k_partidas: int,
        seed_base: int,
        opponents_mode: str = "pool_fijo",
        n_workers: int = 1,
    ) -> list:
        """
        Evalúa todos los individuos de la población y devuelve sus valores
        de fitness.  Si ``n_workers > 1``, la evaluación se paraleliza con
        ``ProcessPoolExecutor``.
        """
        args_list = [
            (genoma, opponents_for_gen, k_partidas, seed_base + i * 9999, opponents_mode)
            for i, genoma in enumerate(poblacion)
        ]

        if n_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                resultados = list(executor.map(_evaluar_individuo_paralelo, args_list))
        else:
            resultados = [_evaluar_individuo_paralelo(args) for args in args_list]

        return [
            self._calcular_fitness_winrate_first(v, pts, rank, n)
            for v, pts, rank, n in resultados
        ]

    def _siguiente_generacion(
        self,
        poblacion: list,
        fitnesses: list,
        sigmas: dict,
        n_elite: int,
        tamano_torneo: int,
        prob_mutacion: float,
        alpha_blx: float,
        rng: random.Random,
    ) -> Tuple[list, dict]:
        """
        Crea la siguiente generación:

        1. Elitismo: conservar los ``n_elite`` mejores individuos intactos.
        2. Descendencia: torneo → BLX-α crossover → mutación adaptativa.

        Devuelve ``(nueva_poblacion, sigmas)`` con sigmas sin modificar
        (el annealing se realiza en :meth:`train`).
        """
        n = len(poblacion)
        ordenados = sorted(range(n), key=lambda i: fitnesses[i], reverse=True)

        # Élites: se copian sin cambios
        nueva_pob = [dict(poblacion[i]) for i in ordenados[:n_elite]]

        # Rellenar con descendencia
        while len(nueva_pob) < n:
            p1 = self._torneo(poblacion, fitnesses, tamano_torneo, rng)
            p2 = self._torneo(poblacion, fitnesses, tamano_torneo, rng)
            h1, h2 = self._blx_alpha_crossover(p1, p2, alpha_blx, rng)
            h1 = self._mutar_adaptativo(h1, sigmas, prob_mutacion, rng)
            h2 = self._mutar_adaptativo(h2, sigmas, prob_mutacion, rng)
            nueva_pob.append(h1)
            if len(nueva_pob) < n:
                nueva_pob.append(h2)

        return nueva_pob, sigmas

    # =========================================================================
    # Entrenamiento principal
    # =========================================================================

    def train(
        self,
        n_generaciones: int = 50,
        n_individuos: int = 30,
        k_partidas: int = 12,
        k_partidas_final: int = 60,
        n_elite: int = 3,
        tamano_torneo: int = 4,
        sigma_mutacion: float = 0.12,
        prob_mutacion: float = 0.25,
        alpha_blx: float = 0.3,
        sigma_min: float = 0.01,
        sigma_decay: float = 0.97,
        sin_mejora_max: int = 15,
        opponents_pool_size: int = None,
        seed: int = 42,
        verbose: bool = True,
        n_workers: int = None,
    ) -> dict:
        """
        Ejecuta el Algoritmo Genético para optimizar los parámetros del agente.

        Al terminar, los mejores parámetros quedan aplicados sobre ``self``
        para que el agente pueda usarse directamente en partidas.

        Parameters
        ----------
        n_generaciones :
            Número máximo de generaciones.
        n_individuos :
            Tamaño de la población.
        k_partidas :
            Partidas por individuo por generación (evaluación estándar).
        k_partidas_final :
            Partidas para la evaluación final del campeón (con todos los oponentes).
        n_elite :
            Individuos élite conservados sin cambios en cada generación.
        tamano_torneo :
            Tamaño del torneo de selección de padres.
        sigma_mutacion :
            Sigma inicial de mutación (fracción del rango de cada gen).
        prob_mutacion :
            Probabilidad de que un gen sea mutado.
        alpha_blx :
            Parámetro de extensión del crossover BLX-α.
        sigma_min :
            Sigma mínimo tras el annealing.
        sigma_decay :
            Factor de decaimiento multiplicativo de sigma por generación.
        sin_mejora_max :
            Generaciones consecutivas sin mejora antes de parar.
        opponents_pool_size :
            Número de oponentes en el pool por generación (``None`` = todos).
        seed :
            Semilla aleatoria para reproducibilidad.
        verbose :
            Mostrar progreso generación a generación.
        n_workers :
            Procesos paralelos para la evaluación (``None`` = CPUs - 1).

        Returns
        -------
        dict
            Los mejores parámetros encontrados.
        """
        # Configurar workers
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 1) - 1)

        # Configurar pool de oponentes
        if opponents_pool_size is None:
            opponents_pool_size = len(_OPPONENT_AGENTS)
        opponents_pool_size = min(opponents_pool_size, len(_OPPONENT_AGENTS))

        rng = random.Random(seed)

        # Inicializar población aleatoria
        poblacion = [_genoma_aleatorio(rng) for _ in range(n_individuos)]

        # Sigmas por gen: todos empiezan con el mismo valor
        sigmas = {nombre: sigma_mutacion for nombre in _GENE_NAMES}

        mejor_fitness_global = -float("inf")
        mejor_genoma_global = dict(poblacion[0])
        generaciones_sin_mejora = 0
        reinicio_realizado_en_ciclo = False
        fitnesses: list = []

        if verbose:
            print(
                f"[GA] Iniciando entrenamiento: {n_generaciones} gen, "
                f"{n_individuos} ind, {k_partidas} partidas/eval, seed={seed}"
            )
            print(
                f"[GA] Pool de oponentes: {opponents_pool_size}/{len(_OPPONENT_AGENTS)}, "
                f"Workers: {n_workers}"
            )

        for gen in range(n_generaciones):
            gen_rng_seed = rng.randint(0, _MAX_SEED)

            # Seleccionar pool de oponentes para esta generación (CRN)
            opponents_for_gen = rng.sample(_OPPONENT_AGENTS, opponents_pool_size)

            # Evaluar población
            fitnesses = self._evaluar_poblacion(
                poblacion,
                opponents_for_gen,
                k_partidas,
                gen_rng_seed,
                opponents_mode="pool_fijo",
                n_workers=n_workers,
            )

            # Mejor individuo de esta generación
            mejor_gen_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            mejor_gen_fitness = fitnesses[mejor_gen_idx]

            if mejor_gen_fitness > mejor_fitness_global:
                mejor_fitness_global = mejor_gen_fitness
                mejor_genoma_global = dict(poblacion[mejor_gen_idx])
                generaciones_sin_mejora = 0
                reinicio_realizado_en_ciclo = False
            else:
                generaciones_sin_mejora += 1

            if verbose:
                avg_fit = sum(fitnesses) / len(fitnesses)
                print(
                    f"[GA] Gen {gen + 1:3d}/{n_generaciones} | "
                    f"Mejor gen: {mejor_gen_fitness:.4f} | "
                    f"Global: {mejor_fitness_global:.4f} | "
                    f"Media: {avg_fit:.4f} | "
                    f"Sin mejora: {generaciones_sin_mejora}"
                )

            # Reinicio parcial del 50% peor cuando llevamos sin_mejora_max//2
            # generaciones sin mejorar (sólo una vez por ciclo de estancamiento)
            if (
                generaciones_sin_mejora == sin_mejora_max // 2
                and not reinicio_realizado_en_ciclo
            ):
                if verbose:
                    print(f"[GA] Reinicio parcial del 50% de la población (gen {gen + 1})")
                poblacion = self._reinicio_parcial(poblacion, fitnesses, 0.5, rng)
                reinicio_realizado_en_ciclo = True
                # Saltamos la creación de descendencia: la próxima generación
                # re-evaluará la población reiniciada
                continue

            # Parada por estancamiento total
            if generaciones_sin_mejora >= sin_mejora_max:
                if verbose:
                    print(
                        f"[GA] Parada por estancamiento tras "
                        f"{sin_mejora_max} generaciones sin mejora"
                    )
                break

            # Crear siguiente generación
            poblacion, sigmas = self._siguiente_generacion(
                poblacion,
                fitnesses,
                sigmas,
                n_elite,
                tamano_torneo,
                prob_mutacion,
                alpha_blx,
                rng,
            )

            # Annealing de sigma: decaimiento global aplicado a todos los genes
            sigma_actual = max(sigma_min, sigma_mutacion * (sigma_decay ** (gen + 1)))
            sigmas = {nombre: sigma_actual for nombre in _GENE_NAMES}

        # ---- Evaluación final con TODOS los oponentes -----------------------
        if verbose:
            print(
                f"\n[GA] Evaluación final con {k_partidas_final} partidas "
                f"y todos los oponentes disponibles..."
            )

        # Candidatos finales: el mejor global + hasta 4 más de la población actual
        candidatos_finales = [mejor_genoma_global]
        for cand in poblacion[:4]:
            cand_copy = dict(cand)
            if cand_copy not in candidatos_finales:
                candidatos_finales.append(cand_copy)
        candidatos_finales = candidatos_finales[:5]

        fitnesses_final = self._evaluar_poblacion(
            candidatos_finales,
            _OPPONENT_AGENTS,   # todos los oponentes en la evaluación final
            k_partidas_final,
            rng.randint(0, _MAX_SEED),
            opponents_mode="pool_fijo",
            n_workers=n_workers,
        )

        mejor_final_idx = max(range(len(fitnesses_final)), key=lambda i: fitnesses_final[i])
        mejor_genoma_final = candidatos_finales[mejor_final_idx]
        mejor_fitness_final = fitnesses_final[mejor_final_idx]

        # Guardar y aplicar los mejores parámetros
        self._mejor_genoma = mejor_genoma_final
        self._mejor_fitness = mejor_fitness_final
        for nombre, valor in mejor_genoma_final.items():
            if hasattr(self, nombre):
                setattr(self, nombre, valor)

        if verbose:
            print(f"[GA] Entrenamiento completado. Mejor fitness final: {mejor_fitness_final:.4f}")
            print("[GA] Mejores parámetros encontrados:")
            for nombre, valor in mejor_genoma_final.items():
                print(f"  {nombre}: {valor}")

        return mejor_genoma_final


# ---------------------------------------------------------------------------
# Smoke-test rápido
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando smoke-test de AgentParametrizableGenetico...")
    agente = AgentParametrizableGenetico(agent_id=0)
    print(f"Agente creado: {agente}")
    resultado = agente.train(
        n_generaciones=3,
        n_individuos=6,
        k_partidas=4,
        k_partidas_final=8,
        verbose=True,
        n_workers=1,
        seed=123,
    )
    print(f"Parámetros encontrados: {resultado}")
    import tempfile
    import os as _os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as _tf:
        tmp = _tf.name
    agente.guardar_params(tmp)
    agente2 = AgentParametrizableGenetico.desde_fichero(tmp, agent_id=1)
    print(f"Agente cargado desde fichero: {agente2}")
    _os.remove(tmp)
    print("Smoke-test completado.")
