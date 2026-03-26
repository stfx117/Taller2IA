from __future__ import annotations

from typing import TYPE_CHECKING
from algorithms.utils import bfs_distance

if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    # TODO: Implement your code here
    
    if state.is_win():
      return 1000
    if state.is_lose():
      return -1000
    
    pos_dron = state.get_drone_position()
    pos_caz = state.get_hunter_positions()
    entregas_pend = list(state.get_pending_deliveries())
    layout = state.get_layout()
    score = state.get_score()
    
    evaluacion_final = score
    
    evaluacion_final -= 100*len(entregas_pend)
    if entregas_pend:
      dist_entregas = []
      # DISTANCIA A PUNTOS DE ENTREGA MÁS CERCANO
      for entrega in entregas_pend:
        dist = bfs_distance(layout, pos_dron, entrega)
        dist_entregas.append(dist)

      min_dist_entrega = min(dist_entregas)
      evaluacion_final += 20/ (min_dist_entrega +1)
      
      for cazador in pos_caz:
        dist__cazador = bfs_distance(layout,cazador, pos_dron, hunter_restricted=True)
      
      if dist__cazador < 2:
        evaluacion_final -= 200 / dist__cazador
      
      evaluacion_final -= 20 * len(entregas_pend)
    return max(min(evaluacion_final, 1000), -1000)
