"""
AgentParametrizable.py
======================
Agente parametrizable para el framework PyCatan.

Para usar con parámetros personalizados, crea una subclase:

    from Agents.AgentParametrizable import AgentParametrizable

    class MiAgente(AgentParametrizable):
        def __init__(self, agent_id):
            super().__init__(agent_id, w_production=0.8, robber_aggression=1.0)

Luego úsala con GameDirector:

    from Managers.GameDirector import GameDirector
    gd = GameDirector(agents=[MiAgente, MiAgente, MiAgente, MiAgente])
    gd.game_start(1)
"""

import random
import sys

from Classes.Constants import (
    MaterialConstants,
    BuildConstants,
    HarborConstants,
    TerrainConstants,
    DevelopmentCardConstants,
)
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Classes.Hand import Hand
from Interfaces.AgentInterface import AgentInterface


# ---------------------------------------------------------------------------
# Pip values: number of dots on the dice token for each probability number.
# Formula: pips = 6 - |7 - p|, except p=0 (desert) -> 0
# ---------------------------------------------------------------------------
_PROB_TO_PIPS = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Maximum possible production score for a single node (3 hexes x 5 pips each)
_MAX_NODE_PRODUCTION = 15.0


class AgentParametrizable(AgentInterface):
    """
    Agente de Catan completamente parametrizable.

    Todos los parámetros tienen valores por defecto y pueden sobreescribirse
    creando una subclase (ver cabecera del módulo).

    Parámetros
    ----------
    w_production : float [0, 1]
        Peso de la producción esperada de recursos al puntuar nodos iniciales.
        Un valor alto prioriza nodos con fichas de dado de alta probabilidad.

    w_diversity : float [0, 1]
        Peso de la diversidad de tipos de recurso al puntuar nodos iniciales.
        Un valor alto prioriza nodos que tocan varios tipos de terreno distintos.

    w_port_access : float [0, 1]
        Peso del acceso a puertos al puntuar nodos iniciales.
        Un valor alto prioriza nodos adyacentes a puertos 2:1 o 3:1.

    city_upgrade_threshold : float [0, 1]
        Multiplicador de puntuación para mejorar pueblos a ciudades.
        Un valor alto hace que el agente prefiera mejorar ciudades antes que construir.

    road_expansion_weight : float [0, 1]
        Prioridad de construcción de carreteras frente a otras estructuras.
        Un valor alto impulsa expandir la red de carreteras.

    robber_aggression : float [0, 1]
        Agresividad al mover el ladrón.
        1.0 = colocar siempre en el hex más productivo del rival.
        0.0 = evitar los propios hexes sin importar el rival.

    trade_acceptance_threshold : float [0, 1]
        Ratio mínimo de ganancia percibida para aceptar una oferta entrante.
        Un valor alto hace al agente más exigente en el comercio.

    dev_card_eagerness : float [0, 1]
        Tendencia a jugar cartas de desarrollo pronto.
        1.0 = jugar siempre que se pueda; 0.0 = guardarlas hasta el final.

    discard_priority : int {0, 1, 2}
        Estrategia al tener más de 7 recursos cuando sale el ladrón.
        0 = descartar aleatoriamente (delega en el GameManager).
        1 = conservar los más necesarios para la próxima construcción.
        2 = conservar los menos producidos (descartar el exceso de los más frecuentes).

    expansion_vs_consolidation : float [0, 1]
        Balance entre expandir (carreteras/nuevos pueblos) y consolidar (ciudades).
        1.0 = priorizar expansión; 0.0 = priorizar consolidación.

    min_road_before_settle : int [0, 5]
        Número mínimo de carreteras construidas antes de priorizar nuevos pueblos.
        Evita construir pueblos sin haber expandido primero la red.

    vp_urgency_threshold : int [6, 9]
        Puntos de victoria a partir de los cuales el agente activa el modo
        de final de partida agresivo (prioriza VP inmediatos).
    """

    def __init__(
        self,
        agent_id: int,
        w_production: float = 0.5,
        w_diversity: float = 0.3,
        w_port_access: float = 0.2,
        city_upgrade_threshold: float = 0.6,
        road_expansion_weight: float = 0.4,
        robber_aggression: float = 0.7,
        trade_acceptance_threshold: float = 0.55,
        dev_card_eagerness: float = 0.5,
        discard_priority: int = 1,
        expansion_vs_consolidation: float = 0.5,
        min_road_before_settle: int = 2,
        vp_urgency_threshold: int = 8,
    ):
        super().__init__(agent_id)

        self.w_production = w_production
        self.w_diversity = w_diversity
        self.w_port_access = w_port_access
        self.city_upgrade_threshold = city_upgrade_threshold
        self.road_expansion_weight = road_expansion_weight
        self.robber_aggression = robber_aggression
        self.trade_acceptance_threshold = trade_acceptance_threshold
        self.dev_card_eagerness = dev_card_eagerness
        self.discard_priority = discard_priority
        self.expansion_vs_consolidation = expansion_vs_consolidation
        self.min_road_before_settle = min_road_before_settle
        self.vp_urgency_threshold = vp_urgency_threshold

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _pip_value(self, probability: int) -> int:
        """Number of pips (dots) for a given dice probability token."""
        return _PROB_TO_PIPS.get(probability, 0)

    # --- Node scoring --------------------------------------------------------

    def _node_production_score(self, node_id: int) -> float:
        """Sum of pip values of all terrains adjacent to a node."""
        total = 0.0
        for terrain_id in self.board.nodes[node_id]['contacting_terrain']:
            terrain = self.board.terrain[terrain_id]
            total += self._pip_value(terrain['probability'])
        return total

    def _node_diversity_score(self, node_id: int) -> float:
        """Fraction of distinct non-desert resource types adjacent to a node (0..1)."""
        types = set()
        for terrain_id in self.board.nodes[node_id]['contacting_terrain']:
            t = self.board.terrain[terrain_id]['terrain_type']
            if t != TerrainConstants.DESERT:
                types.add(t)
        return len(types) / 5.0

    def _node_port_score(self, node_id: int) -> float:
        """1.0 for a 2:1 specific harbor, 0.6 for 3:1 ALL harbor, 0.0 otherwise."""
        harbor = self.board.nodes[node_id]['harbor']
        if harbor == HarborConstants.NONE:
            return 0.0
        if harbor == HarborConstants.ALL:
            return 0.6
        return 1.0

    def _score_node(self, node_id: int) -> float:
        """Weighted composite score for a node as a settlement position."""
        prod_norm = min(self._node_production_score(node_id) / _MAX_NODE_PRODUCTION, 1.0)
        diversity = self._node_diversity_score(node_id)
        port = self._node_port_score(node_id)
        return (
            self.w_production * prod_norm
            + self.w_diversity * diversity
            + self.w_port_access * port
        )

    # --- Resource analysis ---------------------------------------------------

    def _my_production_by_resource(self) -> dict:
        """
        Returns {material_id: total_expected_pips} for all resources
        produced by this player's settlements and cities.
        """
        production = {i: 0 for i in range(5)}
        for node in self.board.nodes:
            if node['player'] == self.id:
                multiplier = 2 if node['has_city'] else 1
                for terrain_id in node['contacting_terrain']:
                    terrain = self.board.terrain[terrain_id]
                    if terrain['terrain_type'] != TerrainConstants.DESERT:
                        # TerrainConstants and MaterialConstants share the same int values
                        resource = terrain['terrain_type']
                        production[resource] += self._pip_value(terrain['probability']) * multiplier
        return production

    def _hand_as_list(self) -> list:
        """Returns current hand resources as a 5-element list [cereal, mineral, clay, wood, wool]."""
        r = self.hand.resources
        return [r.cereal, r.mineral, r.clay, r.wood, r.wool]

    def _resource_surplus_sorted(self) -> list:
        """Returns [(material_id, amount), ...] sorted descending by amount in hand."""
        hand = self._hand_as_list()
        return sorted(enumerate(hand), key=lambda x: x[1], reverse=True)

    def _most_needed_resource(self) -> int:
        """
        Returns the material_id the player needs most,
        based on lowest (hand_amount + production_score).
        """
        production = self._my_production_by_resource()
        hand = self._hand_as_list()
        scores = [(i, hand[i] + production[i] * 0.1) for i in range(5)]
        scores.sort(key=lambda x: x[1])
        return scores[0][0]

    def _most_surplus_resource(self) -> int:
        """Returns the material_id this player has most of in hand."""
        return self._resource_surplus_sorted()[0][0]

    def _resources_needed_for(self, building: str) -> list:
        """Returns [(material_id, deficit), ...] for resources missing to build `building`."""
        costs = {
            BuildConstants.TOWN: (1, 0, 1, 1, 1),
            BuildConstants.CITY: (2, 3, 0, 0, 0),
            BuildConstants.ROAD: (0, 0, 1, 1, 0),
            BuildConstants.CARD: (1, 1, 0, 0, 1),
        }
        hand = self._hand_as_list()
        cost = costs.get(building, (0, 0, 0, 0, 0))
        return [(i, cost[i] - hand[i]) for i in range(5) if cost[i] - hand[i] > 0]

    # --- Victory points & game state ----------------------------------------

    def _current_victory_points(self) -> int:
        """
        Estimate current VP from buildings on board plus VP development cards in hand.
        Does not account for largest army / longest road (unknown without full game state).
        """
        vp = 0
        for node in self.board.nodes:
            if node['player'] == self.id:
                vp += 2 if node['has_city'] else 1
        vp_cards = self.development_cards_hand.find_card_by_effect(
            DevelopmentCardConstants.VICTORY_POINT_EFFECT
        )
        vp += len(vp_cards)
        return vp

    def _is_endgame(self) -> bool:
        return self._current_victory_points() >= self.vp_urgency_threshold

    def _count_my_roads(self) -> int:
        """Count unique road segments belonging to this player."""
        seen = set()
        for node in self.board.nodes:
            for road in node['roads']:
                if road['player_id'] == self.id:
                    edge = (min(node['id'], road['node_id']), max(node['id'], road['node_id']))
                    seen.add(edge)
        return len(seen)

    # --- Robber helpers ------------------------------------------------------

    def _opponent_production_on_terrain(self, terrain_id: int) -> float:
        """Total expected pip production of opponents on a terrain hex."""
        terrain = self.board.terrain[terrain_id]
        pips = self._pip_value(terrain['probability'])
        total = 0.0
        for node_id in terrain['contacting_nodes']:
            node = self.board.nodes[node_id]
            if node['player'] not in (-1, self.id):
                multiplier = 2 if node['has_city'] else 1
                total += pips * multiplier
        return total

    def _i_produce_on_terrain(self, terrain_id: int) -> bool:
        """True if this player has a settlement or city on the terrain hex."""
        terrain = self.board.terrain[terrain_id]
        return any(self.board.nodes[n]['player'] == self.id for n in terrain['contacting_nodes'])

    # --- Road helpers --------------------------------------------------------

    def _pick_best_road(self, valid_roads: list) -> dict:
        """
        From a list of valid road positions, return the one whose endpoint
        scores highest as a future settlement location (or leads toward one).
        """
        if not valid_roads:
            return None

        best_score = -1.0
        best_road = valid_roads[0]

        for road in valid_roads:
            finish = road['finishing_node']
            score = self._score_node(finish)
            # Small bonus for each adjacent unoccupied node reachable from this road end
            for adj in self.board.nodes[finish]['adjacent']:
                if self.board.nodes[adj]['player'] == -1:
                    score += self._score_node(adj) * 0.3
                    # Additional bonus for nodes two steps away (expansion potential)
                    for adj2 in self.board.nodes[adj]['adjacent']:
                        if self.board.nodes[adj2]['player'] == -1 and adj2 != finish:
                            score += self._score_node(adj2) * 0.1
            if score > best_score:
                best_score = score
                best_road = road

        return best_road

    # --- Dev card selection --------------------------------------------------

    def _choose_dev_card_to_play(self):
        """
        Selects which development card to play given context.
        Returns a DevelopmentCard or None.
        Priority order:
          1. VP card if in endgame
          2. Knight if robber is threatening our terrains
          3. Any non-VP card
          4. VP card (only in endgame, already handled above)
        """
        hand = self.development_cards_hand.hand
        if not hand:
            return None

        # Endgame: play VP cards immediately
        if self._is_endgame():
            vp_cards = self.development_cards_hand.find_card_by_effect(
                DevelopmentCardConstants.VICTORY_POINT_EFFECT
            )
            if vp_cards:
                return vp_cards[0]

        # Robber on our territory: prefer Knight
        for terrain in self.board.terrain:
            if terrain['has_thief'] and self._i_produce_on_terrain(terrain['id']):
                knights = self.development_cards_hand.find_card_by_effect(
                    DevelopmentCardConstants.KNIGHT_EFFECT
                )
                if knights:
                    return knights[0]

        # Play first non-VP card (save VP cards for endgame reveal)
        for card in hand:
            if card.effect != DevelopmentCardConstants.VICTORY_POINT_EFFECT:
                return card

        # Only VP cards remain: play only if endgame
        if self._is_endgame():
            return hand[0]

        return None

    # =========================================================================
    # AgentInterface method implementations
    # =========================================================================

    # --- Initial placement ---------------------------------------------------

    def on_game_start(self, board_instance):
        """
        Place initial settlement and adjacent road.
        Settlement: highest-scoring valid starting node (production + diversity + port).
        Road: toward the second-best unoccupied node from the chosen position.
        """
        self.board = board_instance
        try:
            valid_nodes = self.board.valid_starting_nodes()
            if not valid_nodes:
                valid_nodes = [n['id'] for n in self.board.nodes if n['player'] == -1]
            if not valid_nodes:
                # Absolute fallback
                return 0, self.board.nodes[0]['adjacent'][0]

            # Score and sort
            scored = sorted(valid_nodes, key=lambda n: self._score_node(n), reverse=True)
            chosen_node = scored[0]

            adjacent = self.board.nodes[chosen_node]['adjacent']
            if not adjacent:
                road_to = 0
            elif len(scored) > 1:
                # Road toward the adjacent node with best score (maximizes future expansion)
                road_to = max(adjacent, key=lambda a: self._score_node(a))
            else:
                road_to = adjacent[0]

            return chosen_node, road_to

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_game_start error: {e}", file=sys.stderr)
            return super().on_game_start(board_instance)

    # --- Turn start / end (dev card play) ------------------------------------

    def on_turn_start(self):
        """
        Optionally play a development card before rolling the dice.
        Probability of playing = dev_card_eagerness.
        """
        try:
            if not self.development_cards_hand.hand:
                return None
            if random.random() < self.dev_card_eagerness:
                return self._choose_dev_card_to_play()
            return None
        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_turn_start error: {e}", file=sys.stderr)
            return None

    def on_turn_end(self):
        """
        Optionally play a development card at the end of turn.
        Uses half the eagerness (already had chance at start).
        """
        try:
            if not self.development_cards_hand.hand:
                return None
            if random.random() < self.dev_card_eagerness * 0.5:
                return self._choose_dev_card_to_play()
            return None
        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_turn_end error: {e}", file=sys.stderr)
            return None

    # --- Robber movement -----------------------------------------------------

    def on_moving_thief(self):
        """
        Place the robber using robber_aggression to blend between:
          - Targeting the hex with highest opponent production (aggression = 1.0)
          - Avoiding own hexes with lowest self-damage (aggression = 0.0)
        Also selects a player to steal from on the chosen hex.
        """
        try:
            # Find current thief location to avoid sending it back
            current_thief_terrain = -1
            for terrain in self.board.terrain:
                if terrain['has_thief']:
                    current_thief_terrain = terrain['id']
                    break

            candidates = []
            for terrain in self.board.terrain:
                if terrain['id'] == current_thief_terrain:
                    continue
                if terrain['terrain_type'] == TerrainConstants.DESERT:
                    continue

                opp_prod = self._opponent_production_on_terrain(terrain['id'])
                own_prod = (
                    self._pip_value(terrain['probability'])
                    if self._i_produce_on_terrain(terrain['id'])
                    else 0
                )
                # Higher score = better robber target
                score = self.robber_aggression * opp_prod - (1.0 - self.robber_aggression) * own_prod
                candidates.append((terrain['id'], score))

            if not candidates:
                return {'terrain': current_thief_terrain, 'player': -1}

            candidates.sort(key=lambda x: x[1], reverse=True)
            chosen_terrain_id = candidates[0][0]

            # Pick opponent to rob on that terrain (prefer richer opponents)
            target_player = -1
            best_hand = -1
            for node_id in self.board.terrain[chosen_terrain_id]['contacting_nodes']:
                node = self.board.nodes[node_id]
                if node['player'] not in (-1, self.id):
                    # We approximate wealth by node count (we can't see opponents' hands)
                    opponent_nodes = sum(
                        1 for n in self.board.nodes if n['player'] == node['player']
                    )
                    if opponent_nodes > best_hand:
                        best_hand = opponent_nodes
                        target_player = node['player']

            return {'terrain': chosen_terrain_id, 'player': target_player}

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_moving_thief error: {e}", file=sys.stderr)
            # Safe fallback: don't move the thief (GM will randomize)
            for terrain in self.board.terrain:
                if terrain['has_thief']:
                    return {'terrain': terrain['id'], 'player': -1}
            return {'terrain': 0, 'player': -1}

    # --- Discard (thief with >7 cards) ---------------------------------------

    def on_having_more_than_7_materials_when_thief_is_called(self):
        """
        Discard resources according to discard_priority:
          0 = return full hand (GameManager discards randomly)
          1 = keep resources most needed for next planned build
          2 = keep resources we produce least (discard the most renewable)
        """
        try:
            if self.hand.get_total() <= 7:
                return self.hand

            if self.discard_priority == 0:
                return self.hand  # Delegate to GameManager
            elif self.discard_priority == 1:
                return self._discard_keep_needed()
            elif self.discard_priority == 2:
                return self._discard_keep_high_production()
            else:
                return self.hand

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] discard error: {e}", file=sys.stderr)
            return self.hand

    def _discard_keep_needed(self) -> Hand:
        """
        Keep resources most needed for the next planned build.
        Discard the least-needed resources first.
        Must end up with floor(total / 2) cards.
        """
        total = self.hand.get_total()
        target = total // 2  # must keep at most this many
        to_discard = total - target

        # Determine what we need for the most valuable affordable build
        build_priority = [BuildConstants.CITY, BuildConstants.TOWN, BuildConstants.CARD, BuildConstants.ROAD]
        need_score = {i: 0 for i in range(5)}
        for building in build_priority:
            deficits = self._resources_needed_for(building)
            if deficits:
                for mat_id, deficit in deficits:
                    need_score[mat_id] += deficit
                break

        kept = list(self._hand_as_list())  # mutable copy

        # Discard from least-needed materials first
        discard_order = sorted(range(5), key=lambda i: need_score[i])
        remaining = to_discard
        for mat_id in discard_order:
            if remaining <= 0:
                break
            can_discard = min(kept[mat_id], remaining)
            kept[mat_id] -= can_discard
            remaining -= can_discard

        new_hand = Hand()
        new_hand.resources = Materials(*kept)
        return new_hand

    def _discard_keep_high_production(self) -> Hand:
        """
        Discard the resources we produce most (they'll come back quickly).
        Keep the scarce / low-production resources.
        """
        total = self.hand.get_total()
        target = total // 2
        to_discard = total - target

        production = self._my_production_by_resource()
        kept = list(self._hand_as_list())

        # Discard most-produced resources first
        discard_order = sorted(range(5), key=lambda i: production[i], reverse=True)
        remaining = to_discard
        for mat_id in discard_order:
            if remaining <= 0:
                break
            can_discard = min(kept[mat_id], remaining)
            kept[mat_id] -= can_discard
            remaining -= can_discard

        new_hand = Hand()
        new_hand.resources = Materials(*kept)
        return new_hand

    # --- Commerce phase ------------------------------------------------------

    def on_commerce_phase(self):
        """
        Decide commerce action each time it is called:
          1. Optionally play a dev card (low probability, scaled by eagerness).
          2. Try a harbor trade (4:1 or better) if we have surplus.
          3. Propose a player-to-player TradeOffer.
          4. Return None to end trading.
        """
        try:
            # Dev card during commerce (rare, but possible)
            if (
                self.development_cards_hand.hand
                and random.random() < self.dev_card_eagerness * 0.25
            ):
                card = self._choose_dev_card_to_play()
                if card:
                    return card

            # Harbor trade
            harbor_trade = self._try_harbor_trade()
            if harbor_trade:
                return harbor_trade

            # Player trade offer
            return self._build_trade_offer()

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_commerce_phase error: {e}", file=sys.stderr)
            return None

    def _try_harbor_trade(self):
        """
        Attempt a harbor (bank) trade: most surplus resource for most needed.
        Returns dict{'gives': int, 'receives': int} or None.
        """
        needed = self._most_needed_resource()
        surplus_list = self._resource_surplus_sorted()

        harbor_map = [
            HarborConstants.CEREAL,
            HarborConstants.MINERAL,
            HarborConstants.CLAY,
            HarborConstants.WOOD,
            HarborConstants.WOOL,
        ]

        for mat_id, amount in surplus_list:
            if mat_id == needed or amount == 0:
                continue
            harbor = self.board.check_for_player_harbors(self.id, harbor_map[mat_id])
            if harbor == harbor_map[mat_id] and amount >= 2:
                return {'gives': mat_id, 'receives': needed}
            elif harbor == HarborConstants.ALL and amount >= 3:
                return {'gives': mat_id, 'receives': needed}
            elif amount >= 4:
                return {'gives': mat_id, 'receives': needed}

        return None

    def _build_trade_offer(self):
        """
        Construct a player-to-player TradeOffer.
        Offer 1 unit of our most surplus resource for 1 unit of our most needed.
        Returns a TradeOffer or None.
        """
        surplus_list = self._resource_surplus_sorted()
        needed = self._most_needed_resource()

        for mat_id, amount in surplus_list:
            if mat_id == needed or amount == 0:
                continue
            gives = Materials.from_ids(mat_id, 1)
            receives = Materials.from_ids(needed, 1)
            return TradeOffer(gives, receives)

        return None

    # --- Incoming trade offer ------------------------------------------------

    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_id=int):
        """
        Evaluate an incoming trade offer from another player.

        offer.gives  = what the offering player sends us (we receive)
        offer.receives = what the offering player wants from us (we give)

        Accept if perceived gain ratio >= trade_acceptance_threshold.
        Counter-offer not implemented (always accept or reject).
        """
        try:
            self.board = board_instance
            production = self._my_production_by_resource()
            hand = self._hand_as_list()

            # Check we can afford what they ask
            for i, amount in enumerate(offer.receives):
                if amount > hand[i]:
                    return False  # Can't pay

            # Value of what we receive: higher if we need it
            receive_value = 0.0
            for i, amount in enumerate(offer.gives):
                if amount > 0:
                    scarcity = 1.0 / (1.0 + hand[i] + production[i] * 0.1)
                    receive_value += amount * scarcity

            # Cost of what we give up: higher if we need to keep it
            give_cost = 0.0
            for i, amount in enumerate(offer.receives):
                if amount > 0:
                    remaining = hand[i] - amount
                    retention_value = 1.0 / (1.0 + remaining + production[i] * 0.1)
                    give_cost += amount * retention_value

            if give_cost < 1e-9:
                return True  # Free resources, always accept

            ratio = receive_value / give_cost
            return ratio >= self.trade_acceptance_threshold

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_trade_offer error: {e}", file=sys.stderr)
            return False

    # --- Build phase ---------------------------------------------------------

    def on_build_phase(self, board_instance):
        """
        Decide what to build this turn.

        Scores each buildable option (city, town, road, dev card) based on
        parameters, current VP, and whether it is endgame.
        Returns the best action dict or None.
        """
        self.board = board_instance
        try:
            # Optionally play dev card during build phase
            if (
                self.development_cards_hand.hand
                and random.random() < self.dev_card_eagerness * 0.2
            ):
                card = self._choose_dev_card_to_play()
                if card:
                    return card

            endgame = self._is_endgame()
            roads_built = self._count_my_roads()
            options = []  # [(score, action_dict), ...]

            # --- City upgrade ---
            if self.hand.resources.has_more(BuildConstants.CITY):
                valid_cities = self.board.valid_city_nodes(self.id)
                if valid_cities:
                    best_city = max(valid_cities, key=lambda n: self._node_production_score(n))
                    score = 3.0 * self.city_upgrade_threshold
                    if endgame:
                        score += 6.0
                    # Consolidation preference boosts cities
                    score *= (1.0 + (1.0 - self.expansion_vs_consolidation))
                    options.append((score, {
                        'building': BuildConstants.CITY,
                        'node_id': best_city,
                    }))

            # --- New settlement ---
            if self.hand.resources.has_more(BuildConstants.TOWN):
                valid_towns = self.board.valid_town_nodes(self.id)
                if valid_towns:
                    best_town = max(valid_towns, key=lambda n: self._score_node(n))
                    score = 2.5
                    if endgame:
                        score += 5.0
                    # Penalise if we haven't built enough roads yet
                    if roads_built < self.min_road_before_settle:
                        score *= 0.25
                    # Expansion preference boosts settlements
                    score *= (1.0 + self.expansion_vs_consolidation * 0.5)
                    options.append((score, {
                        'building': BuildConstants.TOWN,
                        'node_id': best_town,
                    }))

            # --- Road ---
            if self.hand.resources.has_more(BuildConstants.ROAD):
                valid_roads = self.board.valid_road_nodes(self.id)
                if valid_roads:
                    best_road = self._pick_best_road(valid_roads)
                    score = self.road_expansion_weight * 2.0 + self.expansion_vs_consolidation
                    if endgame:
                        score *= 0.4  # Roads rarely win the game alone
                    options.append((score, {
                        'building': BuildConstants.ROAD,
                        'node_id': best_road['starting_node'],
                        'road_to': best_road['finishing_node'],
                    }))

            # --- Development card ---
            if self.hand.resources.has_more(BuildConstants.CARD):
                score = 1.2
                if endgame:
                    score += 2.5  # Could be a VP card
                options.append((score, {'building': BuildConstants.CARD}))

            if not options:
                return None

            # Return highest-scoring action
            options.sort(key=lambda x: x[0], reverse=True)
            return options[0][1]

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_build_phase error: {e}", file=sys.stderr)
            return None

    # --- Development card effect callbacks -----------------------------------

    def on_monopoly_card_use(self):
        """Steal the resource we need most from all other players."""
        try:
            return self._most_needed_resource()
        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_monopoly_card_use error: {e}", file=sys.stderr)
            return random.randint(0, 4)

    def on_road_building_card_use(self):
        """Build two roads toward the best reachable positions."""
        try:
            valid_nodes = self.board.valid_road_nodes(self.id)

            if len(valid_nodes) > 1:
                road1 = self._pick_best_road(valid_nodes)
                remaining = [r for r in valid_nodes if r != road1]
                road2 = self._pick_best_road(remaining) if remaining else None
                return {
                    'node_id': road1['starting_node'],
                    'road_to': road1['finishing_node'],
                    'node_id_2': road2['starting_node'] if road2 else None,
                    'road_to_2': road2['finishing_node'] if road2 else None,
                }
            elif len(valid_nodes) == 1:
                return {
                    'node_id': valid_nodes[0]['starting_node'],
                    'road_to': valid_nodes[0]['finishing_node'],
                    'node_id_2': None,
                    'road_to_2': None,
                }
            return None

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_road_building_card_use error: {e}", file=sys.stderr)
            return None

    def on_year_of_plenty_card_use(self):
        """Pick the two resources we need most."""
        try:
            production = self._my_production_by_resource()
            hand = self._hand_as_list()
            # Sort by (have + expected_income): lowest = most needed
            scores = sorted(range(5), key=lambda i: hand[i] + production[i] * 0.5)
            mat1 = scores[0]
            mat2 = scores[1] if len(scores) > 1 else scores[0]
            return {'material': mat1, 'material_2': mat2}

        except Exception as e:
            print(f"[AgentParametrizable P{self.id}] on_year_of_plenty_card_use error: {e}", file=sys.stderr)
            return {'material': random.randint(0, 4), 'material_2': random.randint(0, 4)}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = AgentParametrizable(agent_id=0)
    print("Agent initialized with default parameters:")
    params = {
        k: v for k, v in agent.__dict__.items()
        if not k.startswith('_') and k not in ('hand', 'board', 'development_cards_hand')
    }
    for k, v in params.items():
        print(f"  {k}: {v}")
