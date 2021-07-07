from pathlib import Path
import tempfile


def test_training_experiment(tmp_path):
    from reward_preprocessing.train_agent import ex

    # for now we just check that it works without errors
    ex.run(config_updates={"run_dir": str(tmp_path), "steps": 10, "num_frames": 10})


def test_dataset_creation(model_path):
    from reward_preprocessing.create_rollouts import ex

    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname) / "dataset"
        ex.run(
            config_updates={
                "model_path": str(model_path),
                "save_path": str(path),
                "num_samples": 10,
            }
        )
