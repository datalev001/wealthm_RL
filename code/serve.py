from pathlib import Path
from time import time

import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import FinanceEnv
from models import Actor, CriticLambda

# ---------- artefacts ----------
DATA_H5   = Path("/app/data/master_timeseries.h5")
ACTOR_WTS = Path("/app/models/actor.pt")
LAMBDA_W  = Path("/app/models/critic_lambda.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- bootstrap ----------
env = FinanceEnv(DATA_H5)
actor     = Actor().to(DEVICE).eval()
critic_l  = CriticLambda().to(DEVICE).eval()
actor.load_state_dict(torch.load(ACTOR_WTS,  map_location=DEVICE))
critic_l.load_state_dict(torch.load(LAMBDA_W, map_location=DEVICE))

lambda_max_val : float = 0.0

# ---------- API ----------
app = FastAPI(title="RL Wealth Planner")

class Req(BaseModel):
    customer_id: int

@app.post("/recommend")
def recommend(req: Req):
    global lambda_max_val
    cid = req.customer_id
    if cid not in env.static.index.get_level_values(0):
        raise HTTPException(404, "Unknown customer_id")

    # ---- forward pass ----
    s0 = env.reset(cid)                              # pandas Series
    s_t = torch.tensor(s0.to_numpy(float)).float().to(DEVICE)[None,None,:]

    t0 = time()
    a   = actor(s_t)[0,0]                            # (-1,1)^4
    λ   = critic_l(s_t, a[None,None,:])[0,0]
    latency_ms = int((time() - t0) * 1000)

    lambda_max_val = float(λ.max())

    # scale to cash amounts
    plan = {
        "delta_equity"    : round(float(a[0]) * 500, 2),
        "delta_bond"      : round(float(a[1]) * 500, 2),
        "delta_gic"       : round(float(a[2]) * 500, 2),
        "prepay_mortgage" : round(float(a[3]) * 200, 2),
    }

    return {
        "customer_id" : cid,
        "plan"        : plan,
        "lambda"      : λ.cpu().tolist(),
        "latency_ms"  : latency_ms,
    }

@app.get("/lambda_max")
def lambda_max():
    return {"lambda_max": lambda_max_val}
