"""
Telegram Alerts for Trading System

Sends trading alerts, status updates, and error notifications
via Telegram bot.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import structlog

logger = structlog.get_logger()

# Try to import telegram
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed")


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    TRADE = "trade"


@dataclass
class AlertConfig:
    """Telegram alert configuration."""

    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    min_level: AlertLevel = AlertLevel.INFO
    rate_limit_seconds: int = 1
    max_queue_size: int = 100


class TelegramAlerts:
    """
    Telegram notification system for trading.

    Features:
    - Trade notifications
    - Error alerts
    - Status updates
    - Rate limiting
    - Message formatting
    """

    def __init__(
        self,
        config: Optional[AlertConfig] = None,
    ):
        """
        Initialize Telegram alerts.

        Args:
            config: Alert configuration.
        """
        self.config = config or AlertConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

        self._bot: Optional[Any] = None
        self._last_send_time: float = 0
        self._message_queue: List[str] = []

        if TELEGRAM_AVAILABLE and self.config.bot_token:
            try:
                self._bot = Bot(token=self.config.bot_token)
                logger.info("Telegram bot initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")

    def is_available(self) -> bool:
        """Check if Telegram is available."""
        return self._bot is not None and bool(self.config.chat_id)

    async def send_message(
        self,
        text: str,
        level: AlertLevel = AlertLevel.INFO,
        parse_mode: str = "HTML",
    ) -> bool:
        """
        Send a message to Telegram.

        Args:
            text: Message text.
            level: Alert level.
            parse_mode: Telegram parse mode (HTML or Markdown).

        Returns:
            True if sent successfully.
        """
        if not self.is_available():
            logger.debug(f"Telegram not available, message: {text[:100]}")
            return False

        if not self.config.enabled:
            return False

        # Check level filter
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL, AlertLevel.TRADE]
        if levels.index(level) < levels.index(self.config.min_level):
            return False

        # Add level emoji
        emoji = self._get_emoji(level)
        formatted_text = f"{emoji} {text}"

        try:
            await self._bot.send_message(
                chat_id=self.config.chat_id,
                text=formatted_text,
                parse_mode=parse_mode,
            )
            logger.debug(f"Telegram message sent: {level}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message_sync(
        self,
        text: str,
        level: AlertLevel = AlertLevel.INFO,
    ) -> bool:
        """Synchronous wrapper for send_message."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Queue for later
                self._message_queue.append((text, level))
                return True
            else:
                return loop.run_until_complete(
                    self.send_message(text, level)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.send_message(text, level))

    def _get_emoji(self, level: AlertLevel) -> str:
        """Get emoji for alert level."""
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.TRADE: "📈",
        }
        return emojis.get(level, "📌")

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    async def send_trade_alert(
        self,
        instrument: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
    ) -> bool:
        """Send trade execution alert."""
        side_emoji = "🟢" if side.lower() == "buy" else "🔴"

        text = f"""
<b>{side_emoji} Trade Executed</b>

Instrument: <code>{instrument}</code>
Side: {side.upper()}
Quantity: {quantity:.4f}
Price: ${price:.2f}
"""
        if pnl is not None:
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            text += f"PnL: {pnl_emoji} ${pnl:.2f}\n"

        return await self.send_message(text, AlertLevel.TRADE)

    async def send_signal_alert(
        self,
        instrument: str,
        signal: int,
        confidence: float,
        votes: Dict[int, int],
    ) -> bool:
        """Send voting signal alert."""
        signal_text = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]
        signal_emoji = {1: "🟢", -1: "🔴", 0: "⚪"}[signal]

        text = f"""
<b>{signal_emoji} Signal Generated</b>

Instrument: <code>{instrument}</code>
Signal: {signal_text}
Confidence: {confidence:.1%}

Votes: Buy={votes.get(1, 0)} | Hold={votes.get(0, 0)} | Sell={votes.get(-1, 0)}
"""
        return await self.send_message(text, AlertLevel.INFO)

    async def send_status_update(
        self,
        equity: float,
        daily_pnl: float,
        positions: int,
        drawdown: float,
    ) -> bool:
        """Send daily status update."""
        pnl_emoji = "✅" if daily_pnl >= 0 else "❌"

        text = f"""
<b>📊 Daily Status Update</b>

Equity: ${equity:,.2f}
Daily P&L: {pnl_emoji} ${daily_pnl:+,.2f}
Positions: {positions}
Drawdown: {drawdown:.1%}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
        return await self.send_message(text, AlertLevel.INFO)

    async def send_error_alert(
        self,
        error_type: str,
        message: str,
        details: Optional[str] = None,
    ) -> bool:
        """Send error alert."""
        text = f"""
<b>❌ Error: {error_type}</b>

{message}
"""
        if details:
            text += f"\n<code>{details[:500]}</code>"

        return await self.send_message(text, AlertLevel.ERROR)

    async def send_circuit_breaker_alert(
        self,
        reason: str,
        consecutive_losses: int,
        cooldown_minutes: int,
    ) -> bool:
        """Send circuit breaker alert."""
        text = f"""
<b>🚨 CIRCUIT BREAKER TRIGGERED</b>

Reason: {reason}
Consecutive Losses: {consecutive_losses}
Cooldown: {cooldown_minutes} minutes

<i>Trading will resume after cooldown period</i>
"""
        return await self.send_message(text, AlertLevel.CRITICAL)

    async def send_startup_notification(
        self,
        version: str = "1.0.0",
        agents_loaded: int = 0,
        instruments: List[str] = None,
    ) -> bool:
        """Send startup notification."""
        instruments_text = ", ".join(instruments[:5]) if instruments else "None"
        if instruments and len(instruments) > 5:
            instruments_text += f" (+{len(instruments) - 5} more)"

        text = f"""
<b>🚀 Trading System Started</b>

Version: {version}
Agents Loaded: {agents_loaded}
Instruments: {instruments_text}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
        return await self.send_message(text, AlertLevel.INFO)

    async def send_shutdown_notification(
        self,
        reason: str = "Manual shutdown",
        final_pnl: Optional[float] = None,
    ) -> bool:
        """Send shutdown notification."""
        text = f"""
<b>🛑 Trading System Shutdown</b>

Reason: {reason}
"""
        if final_pnl is not None:
            pnl_emoji = "✅" if final_pnl >= 0 else "❌"
            text += f"Final P&L: {pnl_emoji} ${final_pnl:+,.2f}\n"

        text += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"

        return await self.send_message(text, AlertLevel.INFO)

    async def process_queue(self) -> int:
        """Process queued messages."""
        if not self._message_queue:
            return 0

        sent = 0
        while self._message_queue:
            text, level = self._message_queue.pop(0)
            if await self.send_message(text, level):
                sent += 1

        return sent


# Singleton instance for easy access
_alerts_instance: Optional[TelegramAlerts] = None


def get_alerts() -> TelegramAlerts:
    """Get or create the alerts instance."""
    global _alerts_instance
    if _alerts_instance is None:
        _alerts_instance = TelegramAlerts()
    return _alerts_instance


def send_alert(
    text: str,
    level: AlertLevel = AlertLevel.INFO,
) -> bool:
    """Quick function to send an alert."""
    return get_alerts().send_message_sync(text, level)
