"""
Two parts:
- convert simple learned rules to human-readable logical implication strings
- show how to enforce a simple rule using OR-Tools CP-SAT (concrete, runnable)
"""
from ortools.sat.python import cp_model

def rule_to_implication(rule_str):
    # simple function: receives strings like "delay > 50" and "priority = 1"
    # here we just return the same string for display
    return rule_str

# Example: enforce rule delay > 50 => priority=1 using OR-Tools
def enforce_rule_delay_priority(delay_value):
    model = cp_model.CpModel()
    # variables
    # delay is a known input (we will test satisfaction)
    # priority is boolean 0/1 variable
    priority = model.NewBoolVar("priority")
    # If delay_value > 50 then priority must be True.
    if delay_value > 50:
        # add constraint priority == 1
        model.Add(priority == 1)
    # Solve to test
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return {"satisfiable": True, "priority_assigned": solver.Value(priority)}
    else:
        return {"satisfiable": False}

