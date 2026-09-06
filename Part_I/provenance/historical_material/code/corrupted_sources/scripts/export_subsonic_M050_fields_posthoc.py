#!/usr/bin/env python3
from pathlib import Path
import inspect
import json
import sys
import torch
import pandas as pd

import src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini as mini

RUNS = [
    Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete"),
    Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach"),
    Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach"),
]

OUT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
OUT.mkdir(parents=True, exist_ok=True)

def find_checkpoint(run):
    for name in ["model_best.pt", "checkpoint_best.pt", "best.pt", "model_final.pt"]:
        p = run / name
        if p.exists():
            return p
    pts = sorted(run.glob("*.pt"))
    if pts:
        return pts[0]
    return None

def load_ckpt(path):
hoc_inventory.logport_subsonic_M050_fields_posthoc.py | tee assets/pinn_subsonic/u_reconstruction_diagnostics_M050/post
^C
^CTraceback (most recent call last):
  File "/lustre/fswork/projects/rech/fdb/usv13rn/These_PINN_KH_RT/code/src/scripts/evaluation/build_subsonic_M050_fields_posthoc.py", line 6, in <module>
    import torch
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/__init__.py", line 1382, in <module>
    from .functional import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/functional.py", line 7, in <module>
    import torch.nn.functional as F
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/nn/__init__.py", line 1, in <module>
    from .modules import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/nn/modules/__init__.py", line 2, in <module>
    from .linear import Identity, Linear, Bilinear, LazyLinear
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/nn/modules/linear.py", line 7, in <module>
    from .. import functional as F
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/nn/functional.py", line 20, in <module>
    from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/_jit_internal.py", line 41, in <module>
    import torch.distributed.rpc
  File "/gpfslocalsup/pub/anaconda-py3/2023.09/envs/pytorch-gpu-2.1.1+py3.11.5/lib/python3.11/site-packages/torch/distributed/rpc/__init__.py", line 70, in <module>
    import torch.distributed.autograd as dist_autograd
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1138, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1078, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1504, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1476, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1631, in find_spec
  File "<frozen importlib._bootstrap_external>", line 161, in _path_isfile
  File "<frozen importlib._bootstrap_external>", line 153, in _path_is_mode_type
  File "<frozen importlib._bootstrap_external>", line 147, in _path_stat
KeyboardInterrupt

(pytorch-gpu-2.1.1+py3.11.5) [usv13rn@jean-zay2: These_PINN_KH_RT]$ ^C
(pytorch-gpu-2.1.1+py3.11.5) [usv13rn@jean-zay2: These_PINN_KH_RT]$ ^C
(pytorch-gpu-2.1.1+py3.11.5) [usv13rn@jean-zay2: These_PINN_KH_RT]$ cd $WORK/These_PINN_KH_RT
set +H

cat > code/src/scripts/evaluation/build_subsonic_M050_fields_posthoc.py <<'PY'
#!/usr/bin/env python3
from pathlib import Path
import inspect
import json
import sys
import torch
import pandas as pd

import src.scripts.training.atlas.direct_pinn.train_kh_subsonic_2d_pressure_pq_firstorder_mini as mini

RUNS = [
    Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete"),
    Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach"),
    Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach"),
]

OUT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
OUT.mkdir(parents=True, exist_ok=True)

def find_checkpoint(run):
    for name in ["model_best.pt", "checkpoint_best.pt", "best.pt", "model_final.pt"]:
        p = run / name
        if p.exists():
            return p
    pts = sorted(run.glob("*.pt"))
    if pts:
        return pts[0]
    return None

def load_ckpt(path):
hoc_inventory.logport_subsonic_M050_fields_posthoc.py | tee assets/pinn_subsonic/u_reconstruction_diagnostics_M050/post
[OK] wrote assets/pinn_subsonic/u_reconstruction_diagnostics_M050/checkpoint_inventory.csv
                                                                       run                                                                               checkpoint                                                                                keys  field_state_dict_nkeys                                                                                                                                                 field_state_dict_sample  ci_state_dict_nkeys                                                                                          ci_state_dict_sample  args_nkeys                                                                                                                                                 args_sample  anchor_df_nkeys            anchor_df_sample
   assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete    assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete/model_best.pt field_state_dict, ci_state_dict, args, anchor_df, best_epoch, best_phase, best_loss                      14 net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias | net.8.weight | net.8.bias | net.10.weight | net.10.bias                    8 net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias          38       output_dir | device | mach_values | alpha_min | alpha_max | train_alphas | anchor_alphas | eval_alphas | ymax | central_ymax | sym_ymax | warm_epochs                4 alpha | Mach | ci | omega_i
     assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach      assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach/model_best.pt field_state_dict, ci_state_dict, args, anchor_df, best_epoch, best_phase, best_loss                      14 net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias | net.8.weight | net.8.bias | net.10.weight | net.10.bias                    8 net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias          39 output_dir | device | mach_values | alpha_min | alpha_max | train_alphas | warm_train_alphas | anchor_alphas | eval_alphas | ymax | central_ymax | sym_ymax                4 alpha | Mach | ci | omega_i
assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach/model_best.pt field_state_dict, ci_state_dict, args, anchor_df, best_epoch, best_phase, best_loss                      14 net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias | net.8.weight | net.8.bias | net.10.weight | net.10.bias                    8 net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias          39 output_dir | device | mach_values | alpha_min | alpha_max | train_alphas | warm_train_alphas | anchor_alphas | eval_alphas | ymax | central_ymax | sym_ymax                4 alpha | Mach | ci | omega_i

=== Diagnostic-like functions found in mini.py ===
plot_complex_pair(y, ref, pred, title, path, ylabel)
run_diagnostics(args, field, ci_net, anchor_df, device)

[INFO] This script only inventories by default because the exact diagnostic function signature must match the local code.
[INFO] Next command below will patch a self-contained exporter if needed.
(pytorch-gpu-2.1.1+py3.11.5) [usv13rn@jean-zay2: These_PINN_KH_RT]$ cd $WORK/These_PINN_KH_RT

cat assets/pinn_subsonic/u_reconstruction_diagnostics_M050/checkpoint_inventory.csv
run,checkpoint,keys,field_state_dict_nkeys,field_state_dict_sample,ci_state_dict_nkeys,ci_state_dict_sample,args_nkeys,args_sample,anchor_df_nkeys,anchor_df_sample
assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete,assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete/model_best.pt,"field_state_dict, ci_state_dict, args, anchor_df, best_epoch, best_phase, best_loss",14,net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias | net.8.weight | net.8.bias | net.10.weight | net.10.bias,8,net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias,38,output_dir | device | mach_values | alpha_min | alpha_max | train_alphas | anchor_alphas | eval_alphas | ymax | central_ymax | sym_ymax | warm_epochs,4,alpha | Mach | ci | omega_i
assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach,assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach/model_best.pt,"field_state_dict, ci_state_dict, args, anchor_df, best_epoch, best_phase, best_loss",14,net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias | net.8.weight | net.8.bias | net.10.weight | net.10.bias,8,net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias,39,output_dir | device | mach_values | alpha_min | alpha_max | train_alphas | warm_train_alphas | anchor_alphas | eval_alphas | ymax | central_ymax | sym_ymax,4,alpha | Mach | ci | omega_i
assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach,assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach/model_best.pt,"field_state_dict, ci_state_dict, args, anchor_df, best_epoch, best_phase, best_loss",14,net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias | net.8.weight | net.8.bias | net.10.weight | net.10.bias,8,net.0.weight | net.0.bias | net.2.weight | net.2.bias | net.4.weight | net.4.bias | net.6.weight | net.6.bias,39,output_dir | device | mach_values | alpha_min | alpha_max | train_alphas | warm_train_alphas | anchor_alphas | eval_alphas | ymax | central_ymax | sym_ymax,4,alpha | Mach | ci | omega_i
(pytorch-gpu-2.1.1+py3.11.5) [usv13rn@jean-zay2: These_PINN_KH_RT]$ cd $WORK/These_PINN_KH_RT
set +H                                                              cd $WORK/These_PINN_KH_RT
set +H
python - <<'PY'
python - <<'PY'port Path
from pathlib import Path
import torch
runs = [
runs = [("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete"),
    Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete"),
    Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach"),ch"),
    Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach"),
]
for run in runs:
for run in runs: "="*120)
    print("\n" + "="*120)
    print(run) / "model_best.pt"
    ckpt = run / "model_best.pt"exists=", ckpt.exists())
    print("checkpoint:", ckpt, "exists=", ckpt.exists())
    if not ckpt.exists():
        continue
    d = torch.load(ckpt, map_location="cpu")
    d = torch.load(ckpt, map_location="cpu")nstance(d, dict) else type(d))
    print("top keys:", list(d.keys()) if isinstance(d, dict) else type(d))
    if isinstance(d, dict):
    if isinstance(d, dict):():
        for k, v in d.items():ict):
            if isinstance(v, dict):ys={len(v)}")
                print(f"\n[{k}] nkeys={len(v)}")
                for kk in list(v.keys())[:40]:
                    vv = v[kk]ple(vv.shape) if hasattr(vv, "shape") else type(vv)
                    shape = tuple(vv.shape) if hasattr(vv, "shape") else type(vv)
                    print(" ", kk, shape)
            else:rint(k, type(v), v if isinstance(v, (int, float, str, bool)) else "")
                print(k, type(v), v if isinstance(v, (int, float, str, bool)) else "")
