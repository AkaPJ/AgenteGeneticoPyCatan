import os
import time
import concurrent.futures
import importlib
import itertools
import csv
import traceback

from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha 
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
from Managers.GameDirector import GameDirector

BENCHMARK_AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]

n_matches_per_permutation = 10 
porcentaje_workers = 0.95

# Agentes a evaluar: (ruta_clase, params)
agentes_a_evaluar = [
    ("Agents.AgentParametrizable.AgentParametrizable", None), # Por ejemplo, si quieres evaluar el agente AdrianHerasAgent, que está en Agents.AdrianHerasAgent sin parámetros adicionales
    # Se pueden poner varios agentes para evaluar y comparar, con y sin parámetros personalizados, por si queremos probar varias configuraciones del mismo agente.
]

def cargar_agente(ruta_clase):
    modulo, clase = ruta_clase.rsplit(".", 1)
    mod = importlib.import_module(modulo)
    return getattr(mod, clase)

def crear_clase_agente_configurada(agente_clase, **kwargs):
    class AgenteConfigurado(agente_clase):
        def __init__(self, agent_id):
            super().__init__(agent_id, **kwargs)

    AgenteConfigurado.__name__ = f"{agente_clase.__name__}_ConfiguradoDict"
    return AgenteConfigurado

def crear_clase_agente_configurada_lista(agente_clase, params_list):
    class AgenteConfigurado(agente_clase):
        def __init__(self, agent_id):
            super().__init__(agent_id, *params_list)

    AgenteConfigurado.__name__ = f"{agente_clase.__name__}_ConfiguradoLista"
    return AgenteConfigurado

def simulate_match(opponents, position, agente_alumno_clase, params=None):
    try:
        if params is not None:
            if isinstance(params, (list, tuple)):
                agente_alumno_class = crear_clase_agente_configurada_lista(agente_alumno_clase, params)
            elif isinstance(params, dict):
                agente_alumno_class = crear_clase_agente_configurada(agente_alumno_clase, **params)
            else:
                raise TypeError("params debe ser lista/tupla o dict")
        else:
            agente_alumno_class = agente_alumno_clase

        match_agents = list(opponents)
        match_agents.insert(position, agente_alumno_class)

        game_director = GameDirector(agents=match_agents, max_rounds=200, store_trace=False)
        game_trace = game_director.game_start(print_outcome=False)

        last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
        last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
        victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

        agent_id = f"J{position}"
        points = int(victory_points[agent_id])
        winner = max(victory_points, key=lambda player: int(victory_points[player]))
        victory = 1 if winner == agent_id else 0

        ordenados = sorted(victory_points.items(), key=lambda item: int(item[1]), reverse=True)
        rank = 4  # Default rank if agent not found
        for idx, (jugador, _) in enumerate(ordenados, start=1):
            if jugador == agent_id:
                rank = idx
                break

        return (victory, points, rank)
    except Exception as e:
        print("Exception:", repr(e))
        print(traceback.format_exc())
        return (0, 0, 4)

if __name__ == '__main__':
    results = {agent+str(params) if params is not None else agent: {'wins': 0, 'points': 0, 'rank_sum': 0} for agent, params in agentes_a_evaluar}

    total_workers = os.cpu_count() or 1
    workers_a_utilizar = max(1, int(total_workers * porcentaje_workers))
    print(f"Workers a utilizar ({porcentaje_workers*100}%): {workers_a_utilizar}")

    start_time = time.time()

    permutations = list(itertools.permutations(BENCHMARK_AGENTS, 3))
    total_matches = len(agentes_a_evaluar) * len(permutations) * 4 * n_matches_per_permutation
    coste_medio_partida_segundos = 0.004
    print(f"Total de partidas a simular: {total_matches}. Tiempo estimado: {total_matches * coste_medio_partida_segundos / 60:.2f} minutos")

    matches_done = 0
    batch_size = 10000
    futures_batch = []
    resumen_csv = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_a_utilizar) as executor:
        def task_generator():
            for agente_path, params in agentes_a_evaluar:
                agente_cls = cargar_agente(agente_path)
                for perm in permutations:
                    for pos in range(4):
                        for _ in range(n_matches_per_permutation):
                            yield (list(perm), pos, agente_cls, params, agente_path)


        for perm, pos, agente_cls, params, agente_path in task_generator():
            fut = executor.submit(simulate_match, perm, pos, agente_cls, params=params)
            futures_batch.append((fut, agente_path+str(params) if params is not None else agente_path))


            if len(futures_batch) >= batch_size:
                futures_dict = {fut: agente_alumno for fut, agente_alumno in futures_batch}
                for fut in concurrent.futures.as_completed(futures_dict):
                    victory, points, rank = fut.result()
                    agent = futures_dict[fut]
                    results[agent]['wins'] += victory
                    results[agent]['points'] += points
                    results[agent]['rank_sum'] += rank
                    matches_done += 1
                    if matches_done % 10000 == 0 or matches_done == total_matches:
                        print(f"Progreso: {matches_done}/{total_matches} partidas completadas ({matches_done/total_matches:.2%})")
                futures_batch = []

        if futures_batch:
            futures_dict = {fut: agente_alumno for fut, agente_alumno in futures_batch}
            for fut in concurrent.futures.as_completed(futures_dict):
                victory, points, rank = fut.result()
                agent = futures_dict[fut]
                results[agent]['wins'] += victory
                results[agent]['points'] += points
                results[agent]['rank_sum'] += rank
                matches_done += 1
                if matches_done % 10000 == 0 or matches_done == total_matches:
                    print(f"Progreso: {matches_done}/{total_matches} partidas completadas ({matches_done/total_matches:.2%})")

    partidas_por_agente = len(permutations) * 4 * n_matches_per_permutation
    print("\nResultados ordenados por ratio de victorias:")

    resumen = []
    for agente, stats in results.items():
        nombre = agente
        wins = stats['wins']
        points = stats['points']
        rank_sum = stats['rank_sum']
        ratio = wins / partidas_por_agente
        avg_points = points / partidas_por_agente
        puesto_medio = rank_sum / partidas_por_agente
        resumen.append((nombre, wins, points, partidas_por_agente, ratio, avg_points, puesto_medio))

    resumen.sort(key=lambda x: x[4], reverse=True)

    for nombre, wins, points, total, ratio, avg_points, puesto_medio in resumen:
        print(f"{nombre}: {wins} victorias, {points} puntos en {total} partidas — "
              f"Ratio: {ratio:.2%}, Media puntos: {avg_points:.2f}, Puesto medio: {puesto_medio:.2f}")
        resumen_csv.append([nombre, wins, points, total, f"{ratio:.4f}", f"{avg_points:.2f}", f"{puesto_medio:.2f}"])

    # Guardar CSV
    csv_filename = "benchmark_vs_estandar_resultados.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Agente", "Victorias", "Puntos", "Partidas", "Ratio Victorias", "Media Puntos", "Puesto Medio"])
        writer.writerows(resumen_csv)

    print(f"\n Resultados guardados en: {csv_filename}")

    end_time = time.time()
    horas, resto = divmod(end_time - start_time, 3600)
    minutos, segundos = divmod(resto, 60)
    print(f"\n Tiempo total: {int(horas)}h {int(minutos)}m {int(segundos)}s")