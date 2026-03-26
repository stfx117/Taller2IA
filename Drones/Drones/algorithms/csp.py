from __future__ import annotations

from typing import TYPE_CHECKING

from collections import deque
import copy

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    return backtracking(csp, {})
  
def backtracking(csp: DroneAssignmentCSP, assignment: dict):
  if csp.is_complete(assignment):
    return assignment

  var = csp.get_unassigned_variables(assignment)[0]
  
  for value in csp.domains[var]:
    
    if csp.is_consistent(var, value, assignment):
      csp.assign(var, value, assignment)
      result = backtracking(csp, assignment)
      
      if result != None:
        return result
      
      csp.unassign(var, assignment)
    
  return None


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    return backtracking_inference(csp, {})
  
def backtracking_inference(csp: DroneAssignmentCSP, assignment: dict):
  if csp.is_complete(assignment):
    return assignment
  
  var = csp.get_unassigned_variables(assignment)[0]
  
  for value in csp.domains[var]:
    
    if csp.is_consistent(var, value, assignment):
      csp.assign(var,value, assignment)
      
      domains_backup = {}
      for v in csp.domains:
        domains_backup[v] = csp.domains[v].copy()
      
      failure = False
      neighbors = csp.get_neighbors(var)
      
      for neighbor in neighbors:
        if neighbor not in assignment:
          domain = csp.domains[neighbor]

          for d in domain.copy():
            if not csp.is_consistent(neighbor, d, assignment):
              domain.remove(d)

          if len(domain) == 0:
            failure = True
            break

      if not failure:
                result = backtracking_inference(csp, assignment)
                if result is not None:
                    return result

      csp.domains = domains_backup

      csp.unassign(var, assignment)

  return None
 
def backtracking_ac3(csp):
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    # TODO: Implement your code here

    domains = copy.deepcopy(csp.domains)

    def ac3(domains, queue):
        while queue:
            xi, xj = queue.popleft()

            if revise(csp, domains, xi, xj):
                if len(domains[xi]) == 0:
                    return False

                for xk in csp.neighbors[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    queue = deque()
    for xi in csp.variables:
        for xj in csp.neighbors[xi]:
            queue.append((xi, xj))

    if not ac3(domains, queue):
        return None

    def backtrack(assignment, domains):

        if len(assignment) == len(csp.variables):
            return assignment

        # escoger primera no asignada
        for v in csp.variables:
            if v not in assignment:
                var = v
                break

        for value in domains[var]:

            if csp.is_consistent(var, value, assignment):

                new_assignment = assignment.copy()
                new_assignment[var] = value

                new_domains = copy.deepcopy(domains)
                new_domains[var] = [value]

                # cola solo con vecinos
                queue = deque()
                for neighbor in csp.neighbors[var]:
                    queue.append((neighbor, var))

                if ac3(new_domains, queue):
                    result = backtrack(new_assignment, new_domains)

                    if result is not None:
                        return result

        return None

    return backtrack({}, domains)   

  
def values_compatible(csp, var_i, val_i, var_j, val_j):
    temp = {var_i: val_i}
    return csp.is_consistent(var_j, val_j, temp)
  
def revise(csp, domains, var_i, var_j):
    eliminado = False
    for val_i in list(domains[var_i]):
        tiene_soporte = False
        for val_j in domains[var_j]:
            if values_compatible(csp, var_i, val_i, var_j, val_j):
                tiene_soporte = True
                break
        if not tiene_soporte:
            domains[var_i].remove(val_i)
            eliminado = True
    return eliminado


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    # TODO: Implement your code here (BONUS)

    domains = copy.deepcopy(csp.domains)

    def select_variable(assignment, domains):

      no_asignadas = []

      for v in csp.variables:
          if v not in assignment:
              no_asignadas.append(v)

      min_d = float("inf")

      for v in no_asignadas:
          tam = len(domains[v])
          if tam < min_d:
              min_d = tam

      candidatos = []

      for v in no_asignadas:
          if len(domains[v]) == min_d:
              candidatos.append(v)

      mejor = None
      max_grado = -1

      for v in candidatos:

          grado = 0

          for n in csp.neighbors[v]:
              if n not in assignment:
                  grado += 1

          if grado > max_grado:
              max_grado = grado
              mejor = v

      return mejor

    def order_values(var, domains, assignment):

      valores = domains[var]
      lista = []

      for val in valores:
          conflictos = csp.get_num_conflicts(var, val, assignment)
          lista.append((val, conflictos))

      for i in range(len(lista)):
          for j in range(i + 1, len(lista)):
              if lista[j][1] < lista[i][1]:
                  temp = lista[i]
                  lista[i] = lista[j]
                  lista[j] = temp

      ordenados = []

      for par in lista:
          ordenados.append(par[0])

      return ordenados
    
    
    def forward_check(var, value, domains, assignment):
        new_domains = copy.deepcopy(domains)

        for neighbor in csp.neighbors[var]:
            if neighbor not in assignment:

                for val in list(new_domains[neighbor]):
                    temp = assignment.copy()
                    temp[var] = value

                    if not csp.is_consistent(neighbor, val, temp):
                        new_domains[neighbor].remove(val)

                # si queda vacío → falla
                if len(new_domains[neighbor]) == 0:
                    return None

        return new_domains

    def backtrack(assignment, domains):

        if len(assignment) == len(csp.variables):
            return assignment

        var = select_variable(assignment, domains)

        for value in order_values(var, domains, assignment):

            if csp.is_consistent(var, value, assignment):

                new_assignment = assignment.copy()
                new_assignment[var] = value

                new_domains = forward_check(var, value, domains, new_assignment)

                if new_domains is not None:
                    result = backtrack(new_assignment, new_domains)

                    if result is not None:
                        return result

        return None

    return backtrack({}, domains)