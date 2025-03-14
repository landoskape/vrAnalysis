from argparse import ArgumentParser
from tqdm import tqdm
from vrAnalysis2.helpers import Timer
from vrAnalysis2.database import get_database
from vrAnalysis2.tracking import Tracker, save_tracker


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_trackers", action="store_true")
    parser.add_argument("--time_loading", action="store_true")
    args = parser.parse_args()

    mousedb = get_database("vrMice")
    tracked_mice = mousedb.get_table(trackerExists=True)["mouseName"].unique()

    if args.save_trackers:
        progress = tqdm(tracked_mice, desc="Adding tracker cache...")
        for mouse in progress:
            progress.set_postfix(mouse=mouse)
            tracker = Tracker(mouse)
            save_tracker(tracker)

    if args.time_loading:
        print("Timing tracker loading...")
        for mouse in tracked_mice:
            with Timer(f"Loading tracker from {mouse}"):
                tracker = Tracker(mouse)
