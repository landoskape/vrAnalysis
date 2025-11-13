from tqdm import tqdm

from vrAnalysis.database import get_database
from dimensionality_manuscript.regression_models.registry import PopulationRegistry

clear_population = False  # Clears population cache
check_exists = True  # Makes only if not existing
force_remake = False  # Remakes even if existing

if __name__ == "__main__":
    sessiondb = get_database("vrSessions")
    registry = PopulationRegistry()
    for ses in tqdm(sessiondb.iter_sessions(imaging=True)):
        if clear_population:
            # Clear if requested
            registry.clear_population(ses)

        if (check_exists and not registry._check_population_exists(ses)) or force_remake:
            # If either checking existence and doesn't yet exist, or forcing to remake, ...
            # ... make the population potentially with a force remake
            _ = registry.get_population(ses, force_remake=force_remake)

        ses.clear_cache()
