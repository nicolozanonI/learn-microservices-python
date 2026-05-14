from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone
import subprocess
import logging
import json
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kedro Trigger API")

class JobStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"

class RunTrainingRequest(BaseModel):
    start_date: str = Field(..., description="ISO 8601 con timezone, es. 2026-05-01T00:00:00+00:00")
    end_date: str = Field(..., description="ISO 8601 con timezone, es. 2026-05-31T23:59:59+00:00")
    pipeline: str = Field("data_science", description="Nome pipeline Kedro (default: training)")

# Task store in-memory (per prod: DB/Redis) [3](https://stackoverflow.com/questions/61836761/get-return-status-from-background-tasks-in-fastapi)[2](https://stackoverflow.com/questions/64901945/how-to-send-a-progress-of-operation-in-a-fastapi-app)
TASK_STORE: dict[str, dict] = {}

@app.post("/run-pipeline", status_code=202)
def run_pipeline(payload: RunTrainingRequest, background_tasks: BackgroundTasks):
    """
    Avvia Kedro in background e ritorna job_id.
    Il client (Streamlit) fa polling su /run-pipeline/{job_id}.
    BackgroundTasks esegue il lavoro dopo la response [1](https://fastapi.tiangolo.com/tutorial/background-tasks/)
    """
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    TASK_STORE[job_id] = {
        "job_id": job_id,
        "status": JobStatus.running,
        "created_at": now,
        "started_at": now,
        "finished_at": None,
        "exit_code": None,
        "error": None,
        "pipeline": payload.pipeline,
        "start_date": payload.start_date,
        "end_date": payload.end_date,
    }

    runtime_params = f"start_date={payload.start_date},end_date={payload.end_date}"
    cmd = ["kedro", "run", "--pipeline", payload.pipeline, "--params", runtime_params]

    def kedro_task():
        logger.info("JOB %s - running: %s", job_id, cmd)
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            TASK_STORE[job_id]["exit_code"] = res.returncode

            if res.returncode == 0:
                TASK_STORE[job_id]["status"] = JobStatus.completed
                TASK_STORE[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
            else:
                TASK_STORE[job_id]["status"] = JobStatus.failed
                TASK_STORE[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                TASK_STORE[job_id]["error"] = {
                    "stdout": res.stdout[-4000:],  # limita per non esplodere la response
                    "stderr": res.stderr[-4000:],
                }
        except Exception as e:
            TASK_STORE[job_id]["status"] = JobStatus.failed
            TASK_STORE[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
            TASK_STORE[job_id]["error"] = {"exception": str(e)}

    background_tasks.add_task(kedro_task)

    return {
        "status": "accepted",
        "job_id": job_id,
        "status_url": f"/run-pipeline/{job_id}",
        "message": "Training pipeline started. Poll status_url until completed/failed.",
    }

@app.get("/run-pipeline/{job_id}")
def get_job_status(job_id: str):
    if job_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="job_id not found")
    return TASK_STORE[job_id]
