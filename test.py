from dm_control import suite
from dm_control import viewer

# Выбираем задачу и окружение (например, cheetah-run)
env = suite.load(domain_name="cheetah", task_name="run")

# Открываем viewer для визуализации
viewer.launch(env)