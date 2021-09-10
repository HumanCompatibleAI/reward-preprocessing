curl -o /root/.mujoco/mjkey.txt "$MUJOCO_KEY_URL"
poetry run pytest -m "not expensive"