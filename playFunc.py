# play.py
import os
import ale_py
import shimmy
import gym
from gym.wrappers.record_video  import RecordVideo
import numpy as np
from typing import Callable, Any


def record_episode(policy: Callable[[np.ndarray], int],
                   video_folder: str = "videos_clase",
                   env_id: str = "ALE/Galaxian-v5",
                   max_steps: int = 1000000) -> str:
    """
    Ejecuta un episodio del entorno ALE/Galaxian-v5 usando la política dada
    y graba el video en MP4, renombrándolo para incluir el puntaje final.
    """

    # Crear carpeta si no existe
    os.makedirs(video_folder, exist_ok=True)

    # Crear entorno con render_mode para capturar frames
    env = gym.make(env_id, render_mode="rgb_array")

    # Envolver con RecordVideo
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,
        name_prefix="galaxian_episode"
    )

    # Reset
    obs, info = env.reset()
    done, truncated = False, False
    total_reward = 0.0
    step_count = 0

    while not (done or truncated) and step_count < max_steps:
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        step_count += 1

    env.close()  # Cierra y permite que Gym escriba el archivo MP4

    print(f"Episodio terminado: pasos={step_count}, puntaje={total_reward}")

    # =========================================================
    #   RENOMBRAR EL VIDEO PARA INCLUIR EL PUNTAJE FINAL
    # =========================================================
    # Buscar el último archivo creado por RecordVideo
    latest_video = None
    latest_time = -1

    for fname in os.listdir(video_folder):
        if fname.endswith(".mp4") and "galaxian_episode" in fname:
            full_path = os.path.join(video_folder, fname)
            mtime = os.path.getmtime(full_path)
            if mtime > latest_time:
                latest_time = mtime
                latest_video = full_path

    if latest_video is None:
        print("⚠️  No se encontró un archivo MP4 generado.")
        return video_folder

    # Crear nombre nuevo con puntaje
    folder, old_name = os.path.split(latest_video)
    base, ext = os.path.splitext(old_name)

    new_name = f"{base}_score_{int(total_reward)}{ext}"
    new_path = os.path.join(folder, new_name)

    # Renombrar archivo
    os.rename(latest_video, new_path)

    print(f"🎥 Video renombrado como: {new_path}")

    return new_path


# --------------------------------------------------------------
# Política aleatoria (solo para pruebas)
# --------------------------------------------------------------
def random_policy(obs: Any) -> int:
    """Política de ejemplo: elige acciones al azar."""
    global _temp_env_for_action_space
    if "_temp_env_for_action_space" not in globals():
        _temp_env_for_action_space = gym.make("ALE/Galaxian-v5")

    return int(_temp_env_for_action_space.action_space.sample())


if __name__ == "__main__":
    # print(f"==>> gym.envs.registration.registry.keys():\n {gym.envs.registration.registry.keys()}")
    record_episode(random_policy)
