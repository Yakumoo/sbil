# sbil
Stable Baselines Imitation Learning

### Installation
```
pip install git+https://github.com/Yakumoo/sbil.git
```

### Quickstart

```python
from stable_baselines3 import SAC
from sbil import adversarial

model = SAC("MlpPolicy", env)
model = adversarial(model, demo_buffer='path/buffer.pkl')
model.learn(total_timesteps=10000)
```
