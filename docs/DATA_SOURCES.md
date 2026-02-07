# Data Sources Inventory

Este documento describe las fuentes de datos existentes en el VPS,
su origen, alcance y limitaciones. No propone cambios ni mejoras.

Ultima actualizacion: 2026-02-07
Responsable: Claude Code (auditoria automatica)

---

## 1. Resumen rapido

| Fuente | Venue | Instrumento | Tipo | Timeframe | Rango | Uso actual |
|--------|-------|-------------|------|-----------|-------|------------|
| NautilusTrader Catalog | BINANCE | BTCUSDT.BINANCE | Bars (OHLCV) | 1-HOUR | 2020-01-01 a 2026-02-03 | Backtest, Training RL, Benchmarking |
| Polygon.io | — | Varios (pendiente) | — | Pendiente | — | No integrado |
| EODHD | — | Varios (pendiente) | — | Pendiente | — | No integrado |

---

## 2. Fuentes de datos detalladas

### 2.1 NautilusTrader Catalog — BTCUSDT.BINANCE

- **Ubicacion en disco**: `/opt/nautilus-agents/data/catalog/`

- **Formato**: ParquetDataCatalog (NautilusTrader nativo)

- **Tamano total**: ~39 MB

- **Proveedor / Origen**:
  - Exchange: BINANCE
  - Descarga via: NautilusTrader (API historica)
  - Generado en: Linux (VPS) — cumple D-019

- **Instrumento**:
  - BTCUSDT.BINANCE

- **Tipo de dato**:
  - Bars (OHLCV)

- **Timeframe**:
  - 1-HOUR

- **Rango temporal**:
  - Desde: 2020-01-01
  - Hasta: 2026-02-03

- **Transformaciones aplicadas**:
  - Ninguna (datos almacenados como bars estandar de Nautilus)

- **Uso actual**:
  - Backtesting
  - Entrenamiento RL
  - Benchmarking Baseline vs ML
  - Evaluacion offline

- **Valido para**:
  - Backtest
  - Paper trading
  - Produccion (pendiente validacion de ejecucion real)

- **Limitaciones conocidas**:
  - Un unico instrumento
  - Periodo incluye distintos regimenes de mercado (bull / bear)

### 2.2 Polygon.io (no activo)

- **Estado**: Disponible en VPS
- **Uso actual**: No integrado
- **Instrumentos**: Varios (pendiente inventario)
- **Timeframes**: Pendiente
- **Notas**:
  - No usado en backtests actuales
  - No usado en paper trading
  - No mezclado con el catalogo Nautilus

### 2.3 EODHD (no activo)

- **Estado**: Disponible en VPS
- **Uso actual**: No integrado
- **Instrumentos**: Varios (pendiente inventario)
- **Timeframes**: Pendiente
- **Notas**:
  - No usado en el sistema actual
  - No validado frente a ejecucion real

---

## 3. Dependencias en el codigo

| Script / Modulo | Datos usados | Comentario |
|-----------------|--------------|------------|
| scripts/train_and_evaluate.py | Nautilus catalog | Split 70/15/15 |
| gym_env/nautilus_env.py | Nautilus catalog | Source of bars |
| scripts/evaluate_policies.py | Nautilus catalog | Evaluacion offline |

---

## 4. Reglas de uso (actuales)

- El catalogo **no debe crearse ni modificarse en Windows**
- El entrenamiento debe ejecutarse **en el VPS**
- No se asume ejecucion realista (slippage / latency) todavia
- Este dataset es la referencia unica actual

---

## 5. Riesgos abiertos relacionados con datos

- Dependencia de un solo instrumento
- Dependencia de un solo timeframe
- Posible sesgo de regimen
- No validado aun contra ejecucion real

---

## 6. Notas

- Este documento es descriptivo, no prescriptivo.
- Cualquier cambio en datos debe reflejarse aqui antes de usarse.
