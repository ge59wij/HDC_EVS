from pathlib import Path
import requests
import tonic
from expelliarmus import Wizard

fpath = Path("/space/chair-nas/tosy/Gen3_Chifoumi_DAT/test/rock_200211_102652_0_0_td.dat")
wizard = Wizard(encoding="evt3")
wizard.set_file(fpath)
arr = wizard.read()
print(f"First event encoded as (t, x, y, p): {arr[0]}")
print(f"Number of events: {len(arr)}.")
print(f"Recording duration: {(arr[-1]['t']-arr[0]['t'])//int(1e6)} s.")


#/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/rock_200211_102813_0_0.h5
#/space/chair-nas/tosy/preprocessed_dat_chifoumi/train/scissor_200211_105502_0_0_td.dat.pkl

