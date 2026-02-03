"""
Agent Selector for Regime-Aware Trading.

Selects and coordinates RL agents based on detected market regime.

Features:
1. Pool of specialized agents by regime
2. Automatic selection based on current regime
3. Ensemble of agents (voting/averaging)
4. Performance tracking by regime

Reference: EVAL-003, sistema_regimen_agentes.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from .regime_detector import RegimeDetector, MarketRegime, RegimeState


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    model_path: str
    regime: MarketRegime
    algorithm: str  # 'PPO', 'SAC', 'TD3', etc.
    strategy_type: str = 'general'  # 'momentum', 'mean_reversion', 'defensive', etc.
    performance: Dict = field(default_factory=dict)
    weight: float = 1.0
    active: bool = True


@dataclass
class AgentPrediction:
    """Prediction from a single agent."""
    agent_name: str
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float
    raw_output: Optional[np.ndarray] = None


@dataclass
class EnsemblePrediction:
    """Combined prediction from multiple agents."""
    action: int
    confidence: float
    agents_used: List[str]
    action_weights: Dict[int, float]
    regime: str
    method: str  # 'best', 'ensemble', 'voting'
    details: Dict = field(default_factory=dict)


@dataclass
class AgentSelectorConfig:
    """Configuration for agent selector."""
    # Selection mode
    selection_mode: str = 'ensemble'  # 'best', 'ensemble', 'voting'
    ensemble_top_n: int = 3

    # Performance tracking
    lookback_performance: int = 20
    min_confidence: float = 0.5

    # Fallback
    fallback_action: int = 0  # Hold if no agent available

    # Regime weights for strategy types
    regime_strategy_weights: Dict = field(default_factory=lambda: {
        'bull_low_vol': {'momentum': 1.0, 'trend': 0.8, 'general': 0.5},
        'bull_high_vol': {'momentum': 0.6, 'trend': 0.8, 'defensive': 0.4, 'general': 0.5},
        'bear_low_vol': {'mean_reversion': 0.8, 'defensive': 0.6, 'general': 0.5},
        'bear_high_vol': {'defensive': 1.0, 'volatility': 0.6, 'general': 0.3},
        'sideways_low_vol': {'mean_reversion': 1.0, 'range': 0.8, 'general': 0.5},
        'sideways_high_vol': {'defensive': 0.8, 'volatility': 0.6, 'general': 0.3},
    })


class AgentSelector:
    """
    Selects and coordinates agents based on market regime.

    Usage:
        selector = AgentSelector()
        selector.register_agent('ppo_bull', 'models/ppo_bull.zip', MarketRegime.BULL_LOW_VOL, 'PPO')
        selector.load_all_agents()

        regime_state = detector.detect(df)
        prediction = selector.predict(observation, regime_state)
    """

    def __init__(self, config: Optional[AgentSelectorConfig] = None):
        """Initialize agent selector."""
        self.config = config or AgentSelectorConfig()
        self.agents: Dict[str, AgentConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.performance_history: List[Dict] = []

    # =========================================================================
    # AGENT REGISTRATION
    # =========================================================================

    def register_agent(
        self,
        name: str,
        model_path: str,
        regime: MarketRegime,
        algorithm: str,
        strategy_type: str = 'general',
        performance: Optional[Dict] = None,
    ) -> None:
        """
        Register an agent in the pool.

        Args:
            name: Unique agent name
            model_path: Path to saved model
            regime: Target regime for this agent
            algorithm: Algorithm type ('PPO', 'SAC', etc.)
            strategy_type: Strategy type ('momentum', 'mean_reversion', etc.)
            performance: Initial performance metrics
        """
        self.agents[name] = AgentConfig(
            name=name,
            model_path=model_path,
            regime=regime,
            algorithm=algorithm,
            strategy_type=strategy_type,
            performance=performance or {},
            weight=1.0,
            active=True,
        )

    def register_agents_from_config(self, config_path: str) -> None:
        """
        Register multiple agents from YAML config file.

        Expected format:
        agents:
          - name: ppo_bull_momentum
            model_path: models/rl/ppo_bull.zip
            regime: bull_low_vol
            algorithm: PPO
            strategy_type: momentum
        """
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for agent_config in config.get('agents', []):
            self.register_agent(
                name=agent_config['name'],
                model_path=agent_config['model_path'],
                regime=MarketRegime(agent_config['regime']),
                algorithm=agent_config['algorithm'],
                strategy_type=agent_config.get('strategy_type', 'general'),
                performance=agent_config.get('performance', {}),
            )

    def register_agents_from_dict(self, agents_list: List[Dict]) -> None:
        """
        Register multiple agents from list of dicts.

        Args:
            agents_list: List of agent configurations
        """
        for agent_config in agents_list:
            regime = agent_config['regime']
            if isinstance(regime, str):
                regime = MarketRegime(regime)

            self.register_agent(
                name=agent_config['name'],
                model_path=agent_config['model_path'],
                regime=regime,
                algorithm=agent_config['algorithm'],
                strategy_type=agent_config.get('strategy_type', 'general'),
                performance=agent_config.get('performance', {}),
            )

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_agent(self, name: str) -> Any:
        """
        Load a specific agent model.

        Args:
            name: Agent name

        Returns:
            Loaded model
        """
        if name in self.loaded_models:
            return self.loaded_models[name]

        if name not in self.agents:
            raise ValueError(f"Agent {name} not registered")

        agent_config = self.agents[name]
        model_path = agent_config.model_path

        # Check if file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load based on algorithm
        try:
            if agent_config.algorithm in ['PPO', 'SAC', 'TD3', 'A2C', 'DQN']:
                from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
                algo_classes = {
                    'PPO': PPO,
                    'SAC': SAC,
                    'TD3': TD3,
                    'A2C': A2C,
                    'DQN': DQN,
                }
                model = algo_classes[agent_config.algorithm].load(model_path)
            else:
                # Custom model - try joblib
                import joblib
                model = joblib.load(model_path)

            self.loaded_models[name] = model
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load agent {name}: {e}")

    def load_all_agents(self) -> int:
        """
        Pre-load all registered agents.

        Returns:
            Number of successfully loaded agents
        """
        loaded = 0
        for name in self.agents:
            try:
                self.load_agent(name)
                loaded += 1
            except Exception as e:
                warnings.warn(f"Failed to load {name}: {e}")

        return loaded

    def unload_agent(self, name: str) -> None:
        """Unload a specific agent to free memory."""
        if name in self.loaded_models:
            del self.loaded_models[name]

    def unload_all_agents(self) -> None:
        """Unload all agents."""
        self.loaded_models.clear()

    # =========================================================================
    # AGENT SELECTION
    # =========================================================================

    def get_agents_for_regime(self, regime: MarketRegime) -> List[AgentConfig]:
        """
        Get agents suitable for a regime.

        Args:
            regime: Target regime

        Returns:
            List of suitable agents
        """
        # Agents specific to this regime
        specific_agents = [
            a for a in self.agents.values()
            if a.regime == regime and a.active
        ]

        if specific_agents:
            return specific_agents

        # Fallback: agents with similar volatility
        current_vol = 'high' if 'high' in regime.value else 'low'
        similar_agents = [
            a for a in self.agents.values()
            if current_vol in a.regime.value and a.active
        ]

        if similar_agents:
            return similar_agents

        # Last resort: all active agents
        return [a for a in self.agents.values() if a.active]

    def select_best_agent(
        self,
        regime: MarketRegime,
    ) -> Tuple[Optional[str], Optional[AgentConfig]]:
        """
        Select the best agent for current regime.

        Args:
            regime: Current market regime

        Returns:
            Tuple of (agent_name, agent_config) or (None, None)
        """
        candidates = self.get_agents_for_regime(regime)

        if not candidates:
            return None, None

        # Sort by recent performance
        def get_score(agent: AgentConfig) -> float:
            sharpe = agent.performance.get('recent_sharpe', 0) or 0
            # Bonus for matching strategy type
            strategy_weights = self.config.regime_strategy_weights.get(regime.value, {})
            strategy_bonus = strategy_weights.get(agent.strategy_type, 0.5)
            return sharpe * strategy_bonus * agent.weight

        candidates.sort(key=get_score, reverse=True)
        best = candidates[0]

        return best.name, best

    def select_ensemble(
        self,
        regime: MarketRegime,
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, AgentConfig, float]]:
        """
        Select ensemble of agents with weights.

        Args:
            regime: Current market regime
            top_n: Number of top agents to select

        Returns:
            List of (agent_name, agent_config, weight)
        """
        if top_n is None:
            top_n = self.config.ensemble_top_n

        candidates = self.get_agents_for_regime(regime)

        if not candidates:
            return []

        # Calculate weights
        weighted_candidates = []
        strategy_weights = self.config.regime_strategy_weights.get(regime.value, {})

        for agent in candidates:
            # Base weight by strategy type
            base_weight = strategy_weights.get(agent.strategy_type, 0.5)

            # Adjust by performance
            perf_mult = 1 + (agent.performance.get('recent_sharpe', 0) or 0) * 0.2

            final_weight = base_weight * perf_mult * agent.weight
            weighted_candidates.append((agent.name, agent, final_weight))

        # Sort and take top N
        weighted_candidates.sort(key=lambda x: x[2], reverse=True)
        top_agents = weighted_candidates[:top_n]

        # Normalize weights
        total_weight = sum(w for _, _, w in top_agents)
        if total_weight > 0:
            top_agents = [(n, a, w / total_weight) for n, a, w in top_agents]

        return top_agents

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(
        self,
        observation: np.ndarray,
        regime_state: RegimeState,
    ) -> EnsemblePrediction:
        """
        Generate prediction based on current regime.

        Args:
            observation: Current observation vector
            regime_state: Current regime state from detector

        Returns:
            EnsemblePrediction with action and metadata
        """
        mode = self.config.selection_mode
        regime = regime_state.regime

        if mode == 'best':
            return self._predict_best(observation, regime)
        elif mode == 'ensemble':
            return self._predict_ensemble(observation, regime)
        elif mode == 'voting':
            return self._predict_voting(observation, regime)
        else:
            raise ValueError(f"Unknown selection mode: {mode}")

    def _predict_best(
        self,
        observation: np.ndarray,
        regime: MarketRegime,
    ) -> EnsemblePrediction:
        """Prediction using best agent."""
        agent_name, agent_config = self.select_best_agent(regime)

        if agent_name is None:
            return EnsemblePrediction(
                action=self.config.fallback_action,
                confidence=0.0,
                agents_used=[],
                action_weights={0: 1.0, 1: 0.0, 2: 0.0},
                regime=regime.value,
                method='fallback',
            )

        # Load and predict
        model = self.load_agent(agent_name)
        action, _states = model.predict(observation, deterministic=True)

        return EnsemblePrediction(
            action=int(action),
            confidence=agent_config.weight,
            agents_used=[agent_name],
            action_weights={int(action): 1.0},
            regime=regime.value,
            method='best',
            details={'selected_agent': agent_name},
        )

    def _predict_ensemble(
        self,
        observation: np.ndarray,
        regime: MarketRegime,
    ) -> EnsemblePrediction:
        """Prediction using weighted ensemble."""
        ensemble = self.select_ensemble(regime)

        if not ensemble:
            return EnsemblePrediction(
                action=self.config.fallback_action,
                confidence=0.0,
                agents_used=[],
                action_weights={0: 1.0, 1: 0.0, 2: 0.0},
                regime=regime.value,
                method='fallback',
            )

        # Get predictions from each agent
        predictions = []
        action_weights = {0: 0.0, 1: 0.0, 2: 0.0}

        for agent_name, agent_config, weight in ensemble:
            try:
                model = self.load_agent(agent_name)
                action, _states = model.predict(observation, deterministic=True)
                action = int(action)

                predictions.append({
                    'agent': agent_name,
                    'action': action,
                    'weight': weight,
                })

                action_weights[action] += weight

            except Exception as e:
                warnings.warn(f"Agent {agent_name} prediction failed: {e}")

        if not predictions:
            return EnsemblePrediction(
                action=self.config.fallback_action,
                confidence=0.0,
                agents_used=[],
                action_weights={0: 1.0, 1: 0.0, 2: 0.0},
                regime=regime.value,
                method='fallback',
            )

        # Final action = highest weighted
        final_action = max(action_weights, key=action_weights.get)
        confidence = action_weights[final_action]

        return EnsemblePrediction(
            action=final_action,
            confidence=confidence,
            agents_used=[p['agent'] for p in predictions],
            action_weights=action_weights,
            regime=regime.value,
            method='ensemble',
            details={'predictions': predictions},
        )

    def _predict_voting(
        self,
        observation: np.ndarray,
        regime: MarketRegime,
    ) -> EnsemblePrediction:
        """Prediction using simple majority voting."""
        ensemble = self.select_ensemble(regime)

        if not ensemble:
            return EnsemblePrediction(
                action=self.config.fallback_action,
                confidence=0.0,
                agents_used=[],
                action_weights={0: 1.0, 1: 0.0, 2: 0.0},
                regime=regime.value,
                method='fallback',
            )

        # Get predictions
        votes = {0: 0, 1: 0, 2: 0}
        agents_used = []

        for agent_name, agent_config, weight in ensemble:
            try:
                model = self.load_agent(agent_name)
                action, _states = model.predict(observation, deterministic=True)
                votes[int(action)] += 1
                agents_used.append(agent_name)
            except Exception as e:
                warnings.warn(f"Agent {agent_name} prediction failed: {e}")

        if not agents_used:
            return EnsemblePrediction(
                action=self.config.fallback_action,
                confidence=0.0,
                agents_used=[],
                action_weights={0: 1.0, 1: 0.0, 2: 0.0},
                regime=regime.value,
                method='fallback',
            )

        # Majority vote
        final_action = max(votes, key=votes.get)
        confidence = votes[final_action] / len(agents_used)

        # Normalize votes to weights
        total_votes = sum(votes.values())
        action_weights = {k: v / total_votes for k, v in votes.items()}

        return EnsemblePrediction(
            action=final_action,
            confidence=confidence,
            agents_used=agents_used,
            action_weights=action_weights,
            regime=regime.value,
            method='voting',
            details={'votes': votes},
        )

    # =========================================================================
    # PERFORMANCE TRACKING
    # =========================================================================

    def update_performance(
        self,
        agent_name: str,
        prediction: EnsemblePrediction,
        actual_return: float,
    ) -> None:
        """
        Update performance metrics for an agent.

        Args:
            agent_name: Agent that made prediction
            prediction: The prediction made
            actual_return: Actual return achieved
        """
        if agent_name not in self.agents:
            return

        # Record
        self.performance_history.append({
            'timestamp': pd.Timestamp.now(),
            'agent': agent_name,
            'action': prediction.action,
            'return': actual_return,
            'regime': prediction.regime,
        })

        # Update agent metrics
        self._update_agent_metrics(agent_name)

    def _update_agent_metrics(self, agent_name: str) -> None:
        """Recalculate metrics for an agent."""
        # Filter history for this agent
        agent_history = [
            h for h in self.performance_history[-self.config.lookback_performance * 10:]
            if h['agent'] == agent_name
        ][-self.config.lookback_performance:]

        if len(agent_history) < 5:
            return

        returns = [h['return'] for h in agent_history]

        # Calculate metrics
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

        win_rate = sum(1 for r in returns if r > 0) / len(returns)

        # Update
        self.agents[agent_name].performance = {
            'recent_sharpe': float(sharpe),
            'recent_return': float(mean_return * 252),
            'win_rate': float(win_rate),
            'n_predictions': len(agent_history),
        }

    def get_agent_rankings(self, regime: Optional[MarketRegime] = None) -> pd.DataFrame:
        """
        Get agent rankings by performance.

        Args:
            regime: Optional regime filter

        Returns:
            DataFrame with agent rankings
        """
        data = []

        for name, agent in self.agents.items():
            if regime is not None and agent.regime != regime:
                continue

            data.append({
                'name': name,
                'regime': agent.regime.value,
                'algorithm': agent.algorithm,
                'strategy_type': agent.strategy_type,
                'sharpe': agent.performance.get('recent_sharpe', 0) or 0,
                'win_rate': agent.performance.get('win_rate', 0) or 0,
                'n_predictions': agent.performance.get('n_predictions', 0) or 0,
                'active': agent.active,
            })

        df = pd.DataFrame(data)

        if len(df) > 0:
            df = df.sort_values('sharpe', ascending=False)

        return df

    def deactivate_agent(self, name: str) -> None:
        """Deactivate an agent."""
        if name in self.agents:
            self.agents[name].active = False

    def activate_agent(self, name: str) -> None:
        """Activate an agent."""
        if name in self.agents:
            self.agents[name].active = True

    def get_stats(self) -> Dict:
        """Get selector statistics."""
        return {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values() if a.active),
            'loaded_models': len(self.loaded_models),
            'performance_records': len(self.performance_history),
            'agents_by_regime': {
                regime.value: sum(1 for a in self.agents.values() if a.regime == regime)
                for regime in MarketRegime
            },
        }
