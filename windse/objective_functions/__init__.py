from os.path import dirname, basename, isfile, join
import glob
import importlib
from pyadjoint import stop_annotating, annotate_tape

### Make sure only the objective dictionary is imported ###
__all__ = ["annotated_objective", "objective_functions", "objective_kwargs"]

### Get all files in folder ###
files = glob.glob(join(dirname(__file__), "*.py"))

### Create a function that will turn off annotation where needed ###
def annotated_objective(objective, *args, **kwargs):
    annotate = annotate_tape(kwargs)
    if annotate:
        out = objective(*args, **kwargs)
    else:
        with stop_annotating():
            out = float(objective(*args, **kwargs))
    return out

### Add the names to a dictionary ###
objective_functions = {}
objective_kwargs = {}
for file in files:

    ### Filter and define which should be imported ###
    if isfile(file) and not basename(file).startswith('_'):
        objective_file = "windse.objective_functions."+basename(file)[:-3]

        ### Try to import the two required elements ###
        try:
            imported_objective = importlib.import_module(objective_file)
            name = imported_objective.name
            objective = imported_objective.objective
            default_kwargs = imported_objective.keyword_defaults
        except:
            raise ValueError("Objective File '"+file+"' is missing name or objective(), check _template_.py for details")

        ### Check to make sure we don't have conflicting names ###
        if name in objective_functions.keys():
            raise ValueError("Two objectives named: "+name+". Please rename one")

        ### If all is good, add the objective to the file
        objective_functions[name] = objective
        objective_kwargs[name] = default_kwargs