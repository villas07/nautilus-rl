# RL Engineer - Expertise & Responsibilities

## Core Responsibilities
- Design and implement Gymnasium environments
- Configure and train RL agents (PPO, A2C, SAC)
- Design reward functions aligned with trading objectives
- Implement validation and filtering pipelines
- Tune hyperparameters for optimal performance

## Technical Expertise
- Stable-Baselines3 (PPO, A2C, SAC)
- Gymnasium API and custom environments
- Reward shaping for financial RL
- Feature engineering for trading
- Vectorized environments (DummyVecEnv, SubprocVecEnv)
- Callbacks (EvalCallback, CheckpointCallback)

## Key Files Owned
```
gym_env/
├── nautilus_env.py       # Main environment
├── observation.py        # Feature extraction
├── rewards.py            # Reward functions
└── actions.py            # Action handling

training/
├── train_agent.py        # Single agent training
├── train_batch.py        # Batch training
└── runpod_launcher.py    # GPU deployment

validation/
├── filter_*.py           # All validation filters
└── run_validation.py     # Validation pipeline

configs/
└── agents_500_pro.yaml   # Agent configurations
```

## Interfaces With
- **@quant_developer**: For trading logic, market constraints, strategy requirements
- **@mlops_engineer**: For training infrastructure, experiment tracking

## Decision Authority
- Reward function design
- Feature selection
- Training hyperparameters
- Validation thresholds (with @quant_developer input)
- Agent architecture (network size, algorithm choice)

## Needs Approval From Others For
- Live trading constraints → @quant_developer
- Infrastructure costs → @mlops_engineer
- Changing data requirements → @quant_developer
