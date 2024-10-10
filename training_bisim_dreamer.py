import warnings
from functools import partial as bind
import os

import dreamerv3
import embodied

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


def main():

  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['dmc_vision'],
      'logdir': f'~/logdir/log_bisim_dreamerv3',
      'run.train_ratio': 32,
      'run.steps' : 1.2e5,
      'dyn.rssm.classes': 0
  })
  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')
  
  def make_agent(config):
    env = make_env(config)
    agent = dreamerv3.Bisim_Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent

  def make_logger(config):
    logdir = embodied.Path(config.logdir)
    os.environ['WANDB_API_KEY'] = 'f834350aba2607b8c5763f9edc8c9f1b47a5d29b'
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        #embodied.logger.TensorBoardOutput(logdir),
        embodied.logger.WandBOutput(logdir.name, config=config),
    ])

  def make_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / 'replay',
        online=config.replay.online)

  def make_env(config, env_id=0):
        import dmc2gym
        from embodied.envs import from_gym
        domain_name = "walker" 
        task_name = "walk" 
        resource_files = "/home/rodya-rad/Desktop/mipt/dreamerv3/idealgas0.mp4" 
        img_source = "video"

        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            resource_files=resource_files,
            img_source=img_source,
            total_frames = 1000,
            seed = 42,
            visualize_reward=False,
            from_pixels=True,
            height=64,
            width=64,
            frame_skip=1
        )

        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, config)
        return env

  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context,
  )

  embodied.run.train(
      bind(make_agent, config),
      bind(make_replay, config),
      bind(make_env, config),
      bind(make_logger, config), args)


if __name__ == '__main__':
  main()
