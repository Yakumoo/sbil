# sbil
Stable Baselines Imitation Learning

### Installation
```
pip install git+https://github.com/Yakumoo/sbil.git
```

### Quickstart

```python
from stable_baselines3 import PPO
from sbil.demo import adversarial
from sbil.data import generate_demo
import gym

env = gym.make('CartPole-v0')
model = PPO("MlpPolicy", env)
model = adversarial(model, demo_buffer=generate_demo(env))
model.learn(total_timesteps=10000)
```

Or using the provided script with `my_config.yaml`:
```yaml
env:
    id: CartPole-v1

learner:
    class: PPO
    policy: MlpPolicy

algorithm:
    demo: adversarial

learn:
    total_timesteps: 10000
```
```shell
python -m still.learn -c path/my_config.yaml
```
