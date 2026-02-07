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
| Polygon.io | — | Ninguno | — | — | — | INACTIVO — sin datos descargados, API key vacia |
| EODHD (CSV raw) | Multiples | 92 instrumentos | Bars (OHLCV) | Daily | ~Feb 2025 a Feb 2026 | No integrado en RL pipeline |
| EODHD (Parquet catalog) | Multiples | 63 archivos (25 ETF/Forex + indices) | Bars (OHLCV) | Daily + 1H | ~1 ano (plan free) | No integrado en RL pipeline |
| EODHD (Nautilus catalog) | Multiples | 68 archivos (ETF + Forex + Crypto) | Bars (OHLCV) | Daily + 1H | ~1 ano (plan free) | No integrado en RL pipeline |
| Binance (en Nautilus catalog) | BINANCE | 6 crypto perpetuals | Bars (OHLCV) | Daily + 1H | Pendiente verificar | Presente en catalog, no usado en RL |
| QuestDB | — | Pendiente inventario | Time-series | — | — | 1.1 GB en disco, no usado en RL pipeline |

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

### 2.2 Polygon.io (INACTIVO — sin datos)

- **Estado**: INACTIVO — codigo existe pero no hay datos descargados
- **Uso actual**: Ninguno
- **API Key en nautilus-agents**: Vacia (`POLYGON_API_KEY=` en `.env`)
- **API Key en deskgrade**: Configurada (proyecto separado `/opt/deskgrade/`)
- **Codigo relacionado** (solo en deskgrade):
  - `/opt/deskgrade/data_providers/polygon_service.py`
  - `/opt/deskgrade/data_providers/polygon_loader.py`
- **Datos en disco**: Ninguno — no se ha descargado ni almacenado data de Polygon
- **Notas**:
  - No usado en backtests, paper trading ni RL pipeline
  - No mezclado con el catalogo Nautilus
  - Para activarlo se necesitaria configurar API key y ejecutar descarga

### 2.3 EODHD — Raw CSV

- **Estado**: ACTIVO — actualizaciones diarias automaticas via cron
- **Ubicacion en disco**: `/opt/nautilus-agents/data/eod/`
- **Formato**: CSV (`timestamp,open,high,low,close,adj_close,volume,warning`)
- **Tamano total**: ~1.9 MB (92 archivos CSV)
- **API Key**: Configurada (`EOD_API_KEY=697e79...` en `.env`)
- **Plan**: FREE — limita historico a ~1 ano
- **Timeframe**: Daily (`_1d.csv`)
- **Rango temporal**: ~Feb 2025 a Feb 2026 (~1 ano, limitado por plan free)
- **Instrumentos** (92 total):

| Region | Exchange | Instrumentos | Cantidad |
|--------|----------|-------------|----------|
| Hong Kong | HK | 0005, 0388, 0700, 0883, 0941, 1299, 1398, 2318, 2628, 9988 | 10 |
| Corea del Sur | KO | 000660, 003550, 005380, 005930, 006400, 028260, 035420, 035720, 051910, 068270 | 10 |
| Alemania | XETRA | ADS, ALV, BAS, BAYN, BMW, DTE, MRK, SAP, SIE, VOW3 | 10 |
| Francia | PA | AI, AIR, BNP, CS, DG, MC, OR, SAN, SU, TTE | 10 |
| UK | LSE | AZN, BATS, BP, DGE, GSK, HSBA, LSEG, RIO, SHEL, ULVR | 10 |
| Paises Bajos | AS | ADYEN, AKZA, ASML, HEIA, INGA, KPN, PHIA, RAND, UNA, WKL | 10 |
| Espana | MC | AMS, BBVA, FER, GRF, IAG, IBE, ITX, REP, SAN, TEF | 10 |
| Suiza | SW | ABBN, GIVN, LONN, NESN, NOVN, ROG, SREN, UBSG, ZURN | 9 |
| Indices | INDX | AEX, AXJO, BSESN, FCHI, GDAXI, HSI, IBEX, KS11, N225, NSEI, SSEC, STOXX50E, TWII | 13 |

- **Actualizacion automatica**: Cron diario (`pipeline_runner.sh daily`), ultima ejecucion 2026-02-07 01:45 UTC (25 simbolos, 360 barras nuevas, 0 errores)
- **Scripts relacionados**:
  - `scripts/download_eodhd_full.py` — descarga historica completa
  - `scripts/update_eodhd_daily.py` — actualizacion incremental diaria
  - `data/adapters/eod_adapter.py` — adaptador runtime
- **Logs**: `/var/log/nautilus-eodhd.log`, `/tmp/eodhd_download.log`
- **Uso actual**: No integrado en RL pipeline
- **Limitaciones conocidas**:
  - Plan free limita historico a ~1 ano
  - 12 activos fallaron en descarga (TSE Japon, NSE India, SS/SZ China, SGX Singapur)

### 2.4 EODHD — Parquet Catalog

- **Ubicacion en disco**: `/opt/nautilus-agents/data/catalog/eodhd/`
- **Formato**: Parquet (catalogo propio, no NautilusTrader nativo)
- **Tamano total**: ~17 MB (63 archivos Parquet)
- **Manifest**: `manifest.json` presente
- **Instrumentos**:

| Tipo | Instrumentos | Timeframes | Cantidad archivos |
|------|-------------|------------|-------------------|
| US ETFs | SPY, QQQ, IWM, DIA, USO, GLD, SLV, TLT, IEF, XLF, XLK, XLE, XLV, XLI, VXX | Daily + 1H | 30 |
| Forex | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP, EURJPY, GBPJPY | Daily + 1H | 20 |
| Indices Asia | N225, HSI, NSEI, KS11, TWII | Daily | 5 |
| Acciones Asia | 0700_HK, 9988_HK, 1299_HK, 0005_HK, 2318_HK, 005930_KO, 000660_KO, 035420_KO | Daily | 8 |

- **Ultima descarga completa**: 2026-02-04 (38 activos ok, 12 fallidos)
- **Total barras en manifest**: 584,666
- **Uso actual**: No integrado en RL pipeline
- **Copia duplicada**: `/opt/nautilus-orchestrator/docker-build/data/catalog/eodhd/` (17 MB, sync Docker)

### 2.5 EODHD — Nautilus Catalog (formato NautilusTrader)

- **Ubicacion en disco**: `/opt/nautilus-agents/data/catalog/data/`
- **Formato**: ParquetDataCatalog (NautilusTrader nativo)
- **Tamano total**: ~21 MB (68 archivos Parquet)
- **Sub-catalogos**:

| Tipo | Instrumentos | Cantidad |
|------|-------------|----------|
| `currency_pair/` | AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY | 10 |
| `equity/` | SPY, QQQ, IWM, DIA, USO, GLD, SLV, TLT, IEF, XLF, XLK, XLE, XLV, XLI, VXX | 15 |
| `bar/` | 25 EODHD daily bars + 12 Binance crypto bars (1D+1H) | 37 |
| `crypto_perpetual/` | BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, XRPUSDT, BNBUSDT | 6 |

- **Scripts de conversion**:
  - `scripts/eodhd_to_catalog.py` — convierte raw EODHD a Parquet catalog
  - `scripts/load_eod_to_catalog.py` — carga a NautilusTrader catalog
  - `scripts/import_eodhd_to_db.py` — importa a QuestDB
- **Uso actual**: No integrado en RL pipeline

### 2.6 Binance Crypto (en Nautilus Catalog)

- **Ubicacion**: Dentro de `/opt/nautilus-agents/data/catalog/data/` (subcarpetas `bar/` y `crypto_perpetual/`)
- **Instrumentos**: BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, XRPUSDT, BNBUSDT
- **Timeframes**: Daily + 1H
- **Formato**: ParquetDataCatalog (NautilusTrader nativo)
- **Notas**:
  - BTCUSDT tambien presente en catalogo principal (seccion 2.1) con rango 2020-2026
  - Los otros 5 crypto perpetuals no estan integrados en RL pipeline
  - Existe copia legacy en `/opt/nautilus-agents/data/catalog/bar/` (368 KB, formato antiguo)

### 2.7 QuestDB (base de datos time-series)

- **Ubicacion en disco**: `/opt/nautilus-agents/data/questdb/`
- **Tamano total**: ~1.1 GB
- **Contenido**: Directorios `conf/`, `db/`, `import/`, `public/`
- **Tipo**: Base de datos time-series (QuestDB)
- **Datos**: Probablemente contiene series temporales importadas via `scripts/import_eodhd_to_db.py`
- **Uso actual**: No integrado en RL pipeline
- **Notas**: No se inspeccionaron tablas (requiere ejecutar QuestDB query)

### 2.8 Catalogos legacy/duplicados

| Ubicacion | Tamano | Contenido |
|-----------|--------|-----------|
| `/opt/nautilus-agents/data/catalog/bar/` | 368 KB | 12 Binance crypto parquets (formato antiguo `data.parquet`) |
| `/opt/nautilus-agents/data/catalog_old/bar/` | 21 MB | Backup antiguo del catalogo |
| `/opt/nautilus-orchestrator/docker-build/data/catalog/eodhd/` | 17 MB | Mirror del catalogo EODHD (sync Docker, 2026-02-05) |

---

## 3. Dependencias en el codigo

| Script / Modulo | Datos usados | Comentario |
|-----------------|--------------|------------|
| scripts/train_and_evaluate.py | Nautilus catalog | Split 70/15/15 |
| gym_env/nautilus_env.py | Nautilus catalog | Source of bars |
| scripts/evaluate_policies.py | Nautilus catalog | Evaluacion offline |
| scripts/download_eodhd_full.py | EODHD API | Descarga historica completa |
| scripts/update_eodhd_daily.py | EODHD API | Actualizacion diaria (cron) |
| scripts/eodhd_to_catalog.py | EODHD CSV raw | Conversion a Parquet catalog |
| scripts/load_eod_to_catalog.py | EODHD Parquet | Carga a NautilusTrader catalog |
| scripts/import_eodhd_to_db.py | EODHD data | Importacion a QuestDB |
| data/adapters/eod_adapter.py | EODHD API | Adaptador runtime |

---

## 4. Reglas de uso (actuales)

- El catalogo **no debe crearse ni modificarse en Windows**
- El entrenamiento debe ejecutarse **en el VPS**
- No se asume ejecucion realista (slippage / latency) todavia
- Este dataset es la referencia unica actual

---

## 5. Riesgos abiertos relacionados con datos

- Dependencia de un solo instrumento (para RL pipeline)
- Dependencia de un solo timeframe (para RL pipeline)
- Posible sesgo de regimen
- No validado aun contra ejecucion real
- EODHD API key en plan FREE — limita historico a ~1 ano
- 12 activos EODHD fallaron en descarga (TSE Japon, NSE India, SS/SZ China, SGX Singapur)
- Datos EODHD existen en 3 formatos duplicados (CSV + Parquet catalog + Nautilus catalog)
- QuestDB (1.1 GB) sin inventario de tablas — contenido no verificado
- Polygon.io sin API key configurada en nautilus-agents — sin datos

---

## 6. Notas

- Este documento es descriptivo, no prescriptivo.
- Cualquier cambio en datos debe reflejarse aqui antes de usarse.
