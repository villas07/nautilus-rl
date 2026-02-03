# Plan de Entrenamiento: 500 Agentes

## Resumen

| Métrica | Valor |
|---------|-------|
| Total agentes | 500 |
| Batches | 63 (8 agentes/batch) |
| Símbolos | 34 |
| Timeframes | 1h, 4h, 1d |
| Algoritmos | PPO, A2C, SAC |

## Distribución por Grupo

| Grupo | Agentes | % |
|-------|---------|---|
| stocks_sectors | 180 | 36% |
| stocks_tech | 126 | 25% |
| crypto_major | 122 | 24% |
| stocks_major | 72 | 14% |

## Estimación de Costes

### GPU: RTX A4000 ($0.17/hr)

| Escenario | Tiempo | Coste |
|-----------|--------|-------|
| 1 batch (8 agentes) | ~2h | $0.34 |
| 10 batches (80 agentes) | ~20h | $3.40 |
| 63 batches (500 agentes) | ~126h | $21.42 |

### GPU: RTX 4090 ($0.44/hr) - Más rápido

| Escenario | Tiempo | Coste |
|-----------|--------|-------|
| 1 batch (8 agentes) | ~1.5h | $0.66 |
| 63 batches (500 agentes) | ~95h | $41.80 |

## Plan de Ejecución

### Fase A: Validación (actual)
- [x] Entrenar batch_2 (6 agentes crypto)
- [ ] Validar con 5 filtros
- [ ] Confirmar calidad de modelos

### Fase B: Escalar Crypto (si Fase A OK)
- [ ] Entrenar batches 0-15 (crypto completo)
- [ ] ~24h, ~$4.00
- [ ] Validar y seleccionar mejores

### Fase C: Escalar Stocks (si Fase B OK)
- [ ] Entrenar batches 16-62 (stocks)
- [ ] ~94h, ~$16.00
- [ ] Validar y seleccionar ensemble final

## Comandos

```bash
# Ver batches disponibles
cat configs/agents_generated/batches.json | jq '.[0:5]'

# Lanzar batch específico
python scripts/launch_batch_training.py --batch batch_000

# Lanzar múltiples batches
python scripts/launch_batch_training.py --batches batch_000 batch_001 batch_002

# Monitorear
./scripts/check_training_status.sh

# Descargar y validar
./scripts/download_models_runpod.sh batch_crypto
python scripts/validate_batch.py --models-dir ./models_batch_crypto
```

## Criterios de Éxito

Según `autonomous_config.yaml`:

| Filtro | Criterio | Esperado |
|--------|----------|----------|
| F1 Basic | Sharpe > 1.2, DD < 15% | ~60% pasan |
| F2 CrossVal | Degradation < 30% | ~70% de F1 |
| F3 Diversity | Correlation < 0.5 | Top 50 |
| F4 WalkForward | 3/4 windows | ~80% de F3 |
| **Final** | | **30-50 agentes** |

## Notas

- Crear catálogo en Linux (regla 3.3)
- Monitor Telegram activo
- Backup modelos antes de cada batch
