curl -o /root/.mujoco/mjkey.txt "$MUJOCO_KEY_URL"
pipenv run pytest -m "not expensive"