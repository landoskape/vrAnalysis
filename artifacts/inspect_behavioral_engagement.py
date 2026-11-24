"""Behavior-only exploration of reward-related engagement within VR sessions."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vrAnalysis.database import get_database
from dimensionality_manuscript.configs.behavior_speed_env import _speed_map

OUTPUT = Path(__file__).parent / "behavioral_engagement_inspection"


def trial_features(session):
    env=np.asarray(session.trial_environment)
    starts=np.asarray(session.loadone("trials.positionTracking"),dtype=int)
    bt=np.asarray(session.loadone("positionTracking.times"))
    pos=np.asarray(session.loadone("positionTracking.position"))
    trial=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(pos)]))
    dt=np.r_[np.diff(bt),0.0]
    dt[np.r_[np.diff(trial)!=0,True]]=0
    length=np.asarray(session.env_length)
    if len(np.unique(length)) != 1:
        raise ValueError("Variable track length")
    length=float(length[0])
    edges=np.linspace(0,length,81)
    centers=(edges[:-1]+edges[1:])/2
    speed=_speed_map(session,edges,-np.inf,1.0)
    reward=np.asarray(session.loadone("trials.rewardPosition"),float)
    half=np.asarray(session.loadone("trials.rewardZoneHalfwidth"),float)
    reward_start=reward-half
    delivered=np.asarray(session.loadone("trials.rewardPositionTracking"),dtype=int)
    lick_idx=np.asarray(session.loadone("licksTracking.positionTracking"),dtype=int)
    assert np.all((lick_idx>=0)&(lick_idx<len(pos)))
    lick_trial=trial[lick_idx]
    # Include the first reward-triggering lick, exclude later consumption licks.
    pre_consumption=(delivered[lick_trial]<0)|(lick_idx<=delivered[lick_trial])
    lick_idx=lick_idx[pre_consumption]
    lick_trial=trial[lick_idx]
    lick_map=np.zeros((len(starts),40),int)
    bins=np.clip(np.floor(pos[lick_idx]/length*40).astype(int),0,39)
    np.add.at(lick_map,(lick_trial,bins),1)
    availability=np.asarray(session.loadone("trials.rewardAvailability"))
    active_lick=np.asarray(session.loadone("trials.activeLicking"))
    active_stop=np.asarray(session.loadone("trials.activeStopping"))
    rows=[]
    for t in range(len(starts)):
        anticipation=(centers>=reward_start[t]-.1*length)&(centers<reward_start[t])
        control=(centers>=reward_start[t]-.225*length)&(centers<reward_start[t]-.125*length)&(centers>=3)
        vpre=float(np.nanmean(speed[t,anticipation])) if np.isfinite(speed[t,anticipation]).sum()>=3 else np.nan
        vcontrol=float(np.nanmean(speed[t,control])) if np.isfinite(speed[t,control]).sum()>=3 else np.nan
        slowdown=(vcontrol-vpre)/(vcontrol+vpre+1e-6) if vcontrol>1 and vpre>=0 else np.nan
        lt=lick_idx[lick_trial==t]
        lp=pos[lt]
        reward_zone=(lp>=reward_start[t]-.05*length)&(lp<reward[t]+half[t])
        opposite_center=(reward[t]-.5*length)%length
        distance=(lp-opposite_center+length/2)%length-length/2
        opposite=(distance>=-half[t]-.05*length)&(distance<half[t])
        nr,no=int(reward_zone.sum()),int(opposite.sum())
        samples=trial==t
        anticip_samples=samples&(pos>=reward_start[t]-.1*length)&(pos<reward_start[t])
        prelick=int(np.sum((lp>=reward_start[t]-.1*length)&(lp<reward_start[t])))
        anticip_duration=float(dt[anticip_samples].sum())
        rows.append(dict(mouse=session.mouse_name,date=str(session.date),session_id=str(session.session_id),trial=t+1,
                         environment=int(env[t]),start_minutes=float((bt[starts[t]]-bt[0])/60),
                         trial_duration=float(dt[samples].sum()),reward_start=float(reward_start[t]),reward_delivered=bool(delivered[t]>=0),
                         reward_available=float(availability[t]),active_licking=bool(active_lick[t]),active_stopping=bool(active_stop[t]),
                         v_approach=vpre,v_control=vcontrol,reward_slowdown=slowdown,mean_speed=float(np.nanmean(speed[t])),
                         anticipatory_licks=prelick,anticipatory_lick_rate=prelick/anticip_duration if anticip_duration>0 else np.nan,
                         any_anticipatory_lick=prelick>0,preconsumption_licks=len(lt),lick_bins=int(np.sum(lick_map[t]>0)),
                         lick_selectivity=(nr-no)/(nr+no) if nr+no else 0.,reward_zone_licks=nr,opposite_zone_licks=no))
    session.clear_cache()
    return pd.DataFrame(rows),dict(speed=speed,lick_map=lick_map,position_centers=centers,trial_environment=env,
                                  reward_start=reward_start,reward_position=reward,reward_halfwidth=half)


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"behavior_maps").mkdir(exist_ok=True)
    sessions=list(get_database("vrSessions").iter_sessions(imaging=True,experimentType="Blender VR"))
    sessions.sort(key=lambda s:(s.mouse_name,str(s.date),str(s.session_id)))
    if not args.all:
        sessions=[s for s in sessions if s.mouse_name=="ATL012" and str(s.date) in ("2023-02-27","2023-02-28","2023-03-01","2023-03-03")]
    tables=[]; inventory=[]; errors=[]
    for s in sessions:
        key=f"{s.mouse_name}_{s.date}_{s.session_id}"
        csv=OUTPUT/"behavior_maps"/f"{key}.csv"
        try:
            if csv.exists():
                table=pd.read_csv(csv)
            else:
                table,maps=trial_features(s)
                table.to_csv(csv,index=False)
                np.savez_compressed(csv.with_suffix(".npz"),**maps)
            tables.append(table)
            row=dict(mouse=s.mouse_name,date=str(s.date),session_id=str(s.session_id),n_trials=len(table),duration_min=float(table.start_minutes.max()),
                     active_licking_fraction=float(table.active_licking.mean()),active_stopping_fraction=float(table.active_stopping.mean()),
                     lick_count=int(table.preconsumption_licks.sum()))
            inventory.append(row)
            print("BEHAVIOR",key,row,flush=True)
            for env,g in table.groupby("environment"):
                chunks=np.array_split(g,3)
                print("TERTILES",env,[{k:round(float(c[k].mean()),3) for k in ("reward_slowdown","mean_speed","any_anticipatory_lick","lick_selectivity","reward_delivered")} for c in chunks],flush=True)
        except Exception as error:
            errors.append(dict(mouse=s.mouse_name,date=str(s.date),session_id=str(s.session_id),error=repr(error)))
            print("ERROR",key,repr(error),flush=True)
    pd.concat(tables,ignore_index=True).to_csv(OUTPUT/("all_trial_behavior.csv" if args.all else "discovery_trial_behavior.csv"),index=False)
    pd.DataFrame(inventory).to_csv(OUTPUT/("session_inventory.csv" if args.all else "discovery_inventory.csv"),index=False)
    pd.DataFrame(errors).to_csv(OUTPUT/"behavior_errors.csv",index=False)


if __name__=="__main__":
    run()
