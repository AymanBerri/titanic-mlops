import mlflow

from src.train import main


def test_train_runs():
    # Run training
    main()

    # Get experiment
    experiment = mlflow.get_experiment_by_name("titanic_baseline")
    assert experiment is not None, "MLflow experiment was not created"

    # Create client and get runs
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0, "No runs found in MLflow experiment"
