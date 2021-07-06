def test_training_experiment():
    from reward_preprocessing.train_agent import ex

    # for now we just check that it works without errors
    ex.run(config_updates={"steps": 10, "num_frames": 10})
