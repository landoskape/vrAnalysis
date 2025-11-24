from tqdm import tqdm

from vrAnalysis.database import get_database
from dimensionality_manuscript.registry import PopulationRegistry

clear_population = False  # Clears population cache
check_exists = False  # Makes only if not existing
force_remake = False  # Remakes even if existing

show_not_existing = True  # Prints sessions that don't exist in the registry

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
            try:
                _ = registry.get_population(ses, force_remake=force_remake)
            except Exception as e:
                print(f"Error creating population for session {ses.session_print()}: {e}")
                continue
            finally:
                registry.clear_population_cache()
                ses.clear_cache()

        if show_not_existing and not registry._check_population_exists(ses):
            print(f"!!!!! Session {ses.session_print()} does not exist in the registry")
