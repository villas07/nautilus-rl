"""
Model Loader for Live Trading

Loads validated RL models for use in live trading.
Manages model lifecycle, caching, and hot-reloading.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time

from stable_baselines3 import PPO, A2C, SAC

import structlog

logger = structlog.get_logger()


@dataclass
class LoadedModel:
    """Container for a loaded model with metadata."""

    agent_id: str
    model: Any
    config: Dict[str, Any]
    loaded_at: float
    symbol: str
    venue: str
    timeframe: str


class ModelLoader:
    """
    Loads and manages RL models for live trading.

    Features:
    - Load validated models
    - Filter by instrument
    - Cache management
    - Hot-reload support
    """

    ALGORITHM_MAP = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
    }

    def __init__(
        self,
        models_dir: str = "/app/models",
        validation_file: Optional[str] = None,
        cache_ttl: int = 3600,  # 1 hour
    ):
        """
        Initialize model loader.

        Args:
            models_dir: Directory containing trained models.
            validation_file: Path to validation results JSON.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.models_dir = Path(models_dir)
        self.validation_file = Path(validation_file) if validation_file else None
        self.cache_ttl = cache_ttl

        self._cache: Dict[str, LoadedModel] = {}
        self._validated_agents: Optional[List[str]] = None

    def get_validated_agents(self) -> List[str]:
        """Get list of validated agent IDs."""
        if self._validated_agents is not None:
            return self._validated_agents

        if self.validation_file and self.validation_file.exists():
            try:
                with open(self.validation_file) as f:
                    data = json.load(f)

                # Look for final agents in validation results
                self._validated_agents = data.get("summary", {}).get("final_agents", [])

                if not self._validated_agents:
                    # Try to find in filter results
                    for filter_name in ["filter_5", "filter_4", "filter_3"]:
                        if filter_name in data:
                            agents = data[filter_name].get("passed_agents", [])
                            if agents:
                                self._validated_agents = agents
                                break

                logger.info(f"Loaded {len(self._validated_agents)} validated agents")
                return self._validated_agents

            except Exception as e:
                logger.error(f"Failed to load validation file: {e}")

        # Fallback: discover all agents
        self._validated_agents = self._discover_agents()
        return self._validated_agents

    def _discover_agents(self) -> List[str]:
        """Discover all available agents."""
        agents = []

        for path in self.models_dir.iterdir():
            if path.is_dir():
                if (path / f"{path.name}_final.zip").exists():
                    agents.append(path.name)
                elif (path / "best" / "best_model.zip").exists():
                    agents.append(path.name)

        return sorted(agents)

    def load_model(self, agent_id: str, force_reload: bool = False) -> Optional[LoadedModel]:
        """
        Load a single model.

        Args:
            agent_id: Agent identifier.
            force_reload: Force reload even if cached.

        Returns:
            LoadedModel or None if not found.
        """
        # Check cache
        if not force_reload and agent_id in self._cache:
            cached = self._cache[agent_id]
            if time.time() - cached.loaded_at < self.cache_ttl:
                return cached

        # Find model file
        model_path = self.models_dir / agent_id / f"{agent_id}_final.zip"
        if not model_path.exists():
            model_path = self.models_dir / agent_id / "best" / "best_model.zip"

        if not model_path.exists():
            logger.warning(f"Model not found: {agent_id}")
            return None

        # Load config
        config = self._load_config(agent_id)
        algorithm = config.get("algorithm", "PPO")

        # Load model
        try:
            AlgorithmClass = self.ALGORITHM_MAP.get(algorithm, PPO)
            model = AlgorithmClass.load(str(model_path))

            loaded = LoadedModel(
                agent_id=agent_id,
                model=model,
                config=config,
                loaded_at=time.time(),
                symbol=config.get("symbol", "SPY"),
                venue=config.get("venue", "IBKR"),
                timeframe=config.get("timeframe", "1h"),
            )

            self._cache[agent_id] = loaded
            logger.info(f"Loaded model: {agent_id}")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load model {agent_id}: {e}")
            return None

    def load_models_for_instrument(
        self,
        instrument_id: str,
    ) -> List[LoadedModel]:
        """
        Load all validated models for a specific instrument.

        Args:
            instrument_id: Instrument ID (e.g., "SPY.IBKR").

        Returns:
            List of loaded models.
        """
        # Parse instrument ID
        parts = instrument_id.split(".")
        symbol = parts[0]
        venue = parts[1] if len(parts) > 1 else None

        # Get validated agents
        validated = self.get_validated_agents()

        # Filter and load matching agents
        models = []
        for agent_id in validated:
            config = self._load_config(agent_id)

            agent_symbol = config.get("symbol", "")
            agent_venue = config.get("venue", "")

            if agent_symbol == symbol:
                if venue is None or agent_venue == venue:
                    loaded = self.load_model(agent_id)
                    if loaded:
                        models.append(loaded)

        logger.info(f"Loaded {len(models)} models for {instrument_id}")
        return models

    def load_all_models(self) -> Dict[str, List[LoadedModel]]:
        """
        Load all validated models grouped by instrument.

        Returns:
            Dict mapping instrument_id to list of models.
        """
        validated = self.get_validated_agents()
        models_by_instrument: Dict[str, List[LoadedModel]] = {}

        for agent_id in validated:
            loaded = self.load_model(agent_id)
            if loaded:
                instrument_id = f"{loaded.symbol}.{loaded.venue}"

                if instrument_id not in models_by_instrument:
                    models_by_instrument[instrument_id] = []

                models_by_instrument[instrument_id].append(loaded)

        return models_by_instrument

    def _load_config(self, agent_id: str) -> Dict[str, Any]:
        """Load agent configuration."""
        config_path = self.models_dir / agent_id / "config.json"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception:
                pass

        # Default config
        return {
            "symbol": "SPY",
            "venue": "IBKR",
            "timeframe": "1h",
            "algorithm": "PPO",
        }

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models."""
        validated = self.get_validated_agents()
        cached = list(self._cache.keys())

        symbol_counts: Dict[str, int] = {}
        venue_counts: Dict[str, int] = {}

        for agent_id in validated:
            config = self._load_config(agent_id)
            symbol = config.get("symbol", "UNKNOWN")
            venue = config.get("venue", "UNKNOWN")

            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            venue_counts[venue] = venue_counts.get(venue, 0) + 1

        return {
            "total_validated": len(validated),
            "currently_cached": len(cached),
            "by_symbol": symbol_counts,
            "by_venue": venue_counts,
        }

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache.clear()
        logger.info("Model cache cleared")

    def reload_all(self) -> None:
        """Reload all cached models."""
        agents_to_reload = list(self._cache.keys())
        self.clear_cache()

        for agent_id in agents_to_reload:
            self.load_model(agent_id, force_reload=True)

        logger.info(f"Reloaded {len(agents_to_reload)} models")


def load_validated_models(
    models_dir: str = "/app/models",
    instrument_id: Optional[str] = None,
    validation_file: Optional[str] = None,
) -> List[Any]:
    """
    Convenience function to load validated models.

    Args:
        models_dir: Directory containing models.
        instrument_id: Optional filter by instrument.
        validation_file: Path to validation results.

    Returns:
        List of model objects (for direct predict() calls).
    """
    loader = ModelLoader(
        models_dir=models_dir,
        validation_file=validation_file,
    )

    if instrument_id:
        loaded_models = loader.load_models_for_instrument(instrument_id)
    else:
        all_models = loader.load_all_models()
        loaded_models = []
        for models in all_models.values():
            loaded_models.extend(models)

    return [lm.model for lm in loaded_models]
