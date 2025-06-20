"""Daily retraining and monitoring for the card-fraud model.

validate_sources ─► build_training_set (Spark) ─► train_and_gate (+promote) ─╮
                                                                            ├─► notify
skew_audit (yesterday's served vectors) ─► drift_report ────────────────────╯

* ``validate_sources`` fails the run on blocking data quality checks, so a
  bad upstream load never reaches a training set.
* ``train_and_gate`` exits non-zero when the promotion gate rejects the
  candidate; ``promote`` then does not run and production is unchanged.
* ``skew_audit`` runs independently of training: it replays yesterday's
  logged online vectors through the offline engine and fails when the
  mismatch rate exceeds the budget.

Tasks run the ``tessera`` image on AKS through KubernetesPodOperator, using
workload identity for storage and Key Vault access; Spark runs on Databricks.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
from airflow.utils.trigger_rule import TriggerRule

IMAGE = Variable.get("tessera_image", default_var="tesseraacr.azurecr.io/tessera:latest")
LAKE = Variable.get("tessera_lake", default_var="abfss://lake@tesseradev.dfs.core.windows.net")
CONFIG = "/etc/tessera/tessera.yaml"
DS = "{{ ds }}"

default_args = {
    "owner": "risk-ml",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
}


def tessera(task_id: str, *args: str, **kwargs: object) -> KubernetesPodOperator:
    return KubernetesPodOperator(
        task_id=task_id,
        name=f"tessera-{task_id.replace('_', '-')}",
        namespace="tessera",
        image=IMAGE,
        cmds=["tessera", "--config", CONFIG],
        arguments=list(args),
        service_account_name="tessera-batch",
        labels={"azure.workload.identity/use": "true"},
        get_logs=True,
        on_finish_action="delete_succeeded_pod",
        **kwargs,
    )


with DAG(
    dag_id="tessera_fraud_daily",
    description="Validate sources, rebuild the training set, retrain, gate, audit skew.",
    schedule="0 3 * * *",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["tessera", "fraud", "ml"],
) as dag:
    validate = tessera("validate_sources", "validate")

    build = DatabricksSubmitRunOperator(
        task_id="build_training_set",
        databricks_conn_id="databricks_default",
        json={
            "run_name": f"tessera-fraud-v1-{DS}",
            "new_cluster": {
                "spark_version": "15.4.x-scala2.12",
                "node_type_id": "Standard_D8ds_v5",
                "autoscale": {"min_workers": 2, "max_workers": 12},
                "spark_env_vars": {"PYTHONPATH": "/Workspace/tessera/src"},
            },
            "libraries": [{"pypi": {"package": "tessera[parquet]"}}],
            "spark_python_task": {
                "python_file": "dbfs:/tessera/pipelines/spark/build_training_set.py",
                "parameters": [
                    "--repo",
                    "/Workspace/tessera/examples/fraud/features.py",
                    "--service",
                    "fraud_v1",
                    "--spine",
                    f"{LAKE}/labels/matured/{DS}/",
                    "--out",
                    f"{LAKE}/training/fraud_v1/{DS}/",
                    "--source",
                    f"transactions={LAKE}/archive/transactions/",
                    "--source",
                    f"card_profiles={LAKE}/cdc/card_profiles/",
                ],
            },
        },
    )

    # Registers the candidate, evaluates the promotion gate against production and
    # promotes only on a pass. A rejected candidate exits 3 and fails the task, so
    # the run is visibly red while production keeps serving the previous version.
    train = tessera(
        "train_and_gate",
        "train",
        "--name",
        "card_fraud",
        "--service",
        "fraud_v1",
        "--dataset",
        f"{LAKE}/training/fraud_v1/{DS}/",
        "--label",
        "is_fraud",
        "--gap",
        "2d",
        "--promote",
    )

    skew = tessera(
        "skew_audit",
        "skew-audit",
        "--service",
        "fraud_v1",
        "--log",
        f"{LAKE}/feature-log/fraud_v1/{{{{ yesterday_ds }}}}.jsonl",
        "--max-mismatch-rate",
        "0.002",
    )

    drift = tessera(
        "drift_report",
        "drift",
        "--name",
        "card_fraud",
        "--log",
        f"{LAKE}/feature-log/fraud_v1/{{{{ yesterday_ds }}}}.jsonl",
    )

    done = EmptyOperator(task_id="notify", trigger_rule=TriggerRule.ALL_DONE)

    validate >> build >> train >> done
    skew >> drift >> done
