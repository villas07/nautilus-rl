# PROJECT LOG — Nautilus Trading System

Este documento mantiene continuidad del proyecto entre sesiones.
No contiene código ni propuestas nuevas, solo hechos, decisiones y estado.

---

## 📅 2026-02-07

### 1) Hechos confirmados
- El baseline determinista (MA20/MA50) es estable y supera a las variantes RL probadas.
- El módulo RL Exit Gate V1 falló (mayor drawdown, política degenerada).
- El ML queda descartado como módulo de ejecución con evidencia.
- El catálogo activo usado por el sistema es:
  - NautilusTrader ParquetDataCatalog
  - BTCUSDT.BINANCE
  - Timeframe: Daily (1-DAY-LAST)
  - Rango: 2021-02-03 a 2026-01-31 (1,793 barras)

### 2) Decisiones tomadas
- Se acepta el baseline determinista como referencia válida.
- Se descarta RL para entradas y salidas.
- Se entra en fase de datos / paper trading / backtest realista.
- Antes de montar nada nuevo, se documentan todas las fuentes de datos existentes.
- Hoy no se reentrena ni se reenvían archivos al VPS.

### 3) Fuentes de datos (estado actual)
- Activas:
  - NautilusTrader catalog (BTCUSDT daily)
- Presentes pero NO integradas:
  - Polygon.io (inactivo, sin datos, API key vacía)
  - EODHD (descargas automáticas, múltiples instrumentos, no integrado en pipelines)
- Ninguna fuente externa alimenta el sistema actual salvo Nautilus.

### 4) Estado técnico
- Repo sincronizado con Git (commit 7c17cef).
- VPS limpio, sin cambios locales pendientes.
- No hay entrenamientos ni ejecuciones en curso.

### 5) Qué NO se ha decidido aún (a propósito)
- Uso futuro de EODHD o Polygon.
- Ampliación de instrumentos o timeframes.
- Elección de motor de backtest final (NautilusTrader vs Lean).
- Inicio de paper trading.

### 6) Próximo paso cuando se retome
- Elegir un único eje operativo:
  - Paper trading con baseline
  - Backtest realista
  - Auditoría de calidad de datos
