from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """
    
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        # TODO: Implement your code here
        actions = state.get_legal_actions(0)
        mejor_action = max(actions, key=lambda a: self.minimax(state.generate_successor(0,a),1, self.depth))
        return mejor_action

    def minimax(self, state: GameState, agent_index, depth):
        if state.is_win() or state.is_lose() or depth == 0:
            return self.evaluation_function(state)
        
        num_agents = state.get_num_agents()
        actions = state.get_legal_actions(agent_index)
        if not actions: 
            return self.evaluation_function(state)
        
        next_agent = (agent_index +1) % num_agents
        next_depth = depth
        if next_agent == 0:
            next_depth = depth-1

        if agent_index ==0:
            valor_mas_alto = float("-inf")
            
            for accion in actions:
                sucesor = state.generate_successor(agent_index, accion)
                
                costo = self.minimax(sucesor, next_agent, next_depth)
                
                if costo > valor_mas_alto:
                    valor_mas_alto = costo
            return valor_mas_alto
        
        else:
            valor_mas_bajo = float("inf")
            
            for accion in actions:
                sucesor = state.generate_successor(agent_index, accion)
                costo = self.minimax(sucesor, next_agent, next_depth)
                
                if costo < valor_mas_bajo:
                    valor_mas_bajo = costo
            return valor_mas_bajo
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        # TODO: Implement your code here (BONUS)
        return None

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """

        def expectimax(estado, profundidad, indice_agente):
            if estado.is_terminal() or profundidad == self.depth:
                return self.evaluation_function(estado)

            numero_agentes = estado.get_num_agents()

            if indice_agente == 0:
                valores = []
                for accion in estado.get_legal_actions(0):
                    sucesor = estado.generate_successor(0, accion)
                    valores.append(expectimax(sucesor, profundidad, 1))
                return max(valores) if valores else self.evaluation_function(estado)
            else:
                acciones = estado.get_legal_actions(indice_agente)
                if not acciones:
                    return self.evaluation_function(estado)

                valores = []
                for accion in acciones:
                    sucesor = estado.generate_successor(indice_agente, accion)

                    siguiente_agente = indice_agente + 1
                    if siguiente_agente == numero_agentes:
                        siguiente_agente = 0
                        siguiente_profundidad = profundidad + 1
                    else:
                        siguiente_profundidad = profundidad

                    valores.append(expectimax(sucesor, siguiente_profundidad, siguiente_agente))

                p = self.prob
                valor_minimo = min(valores)
                valor_promedio = sum(valores) / len(valores)

                return (1 - p) * valor_minimo + p * valor_promedio

        mejor_valor = float('-inf')
        mejor_accion = None

        for accion in state.get_legal_actions(0):
            sucesor = state.generate_successor(0, accion)
            valor = expectimax(sucesor, 0, 1)

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

        return mejor_accion