from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines()

    # training = la tua pipeline data_science
    training = pipelines["data_science"]

    # default = somma di alcune pipeline (solo se esistono)
    names = ("data_processing", "data_science", "reporting")
    selected = [pipelines[name] for name in names if name in pipelines]

    # somma robusta (serve un Pipeline([]) come valore iniziale)
    default_pipeline = sum(selected, Pipeline([]))

    return {
        "training": training,
        "__default__": default_pipeline,
        # (opzionale) se vuoi esporre anche le singole:
        # "data_processing": pipelines.get("data_processing", Pipeline([])),
        # "data_science": pipelines.get("data_science", Pipeline([])),
        # "reporting": pipelines.get("reporting", Pipeline([])),
    }