# GOVERNANCE EVALUATION: EVAL-001

## Documento Evaluado
- **Archivo:** `sistema_ml_institucional.md`
- **Origen:** Documentación ML institucional (Hedge Fund / Prop Trading)
- **Fecha evaluación:** 2026-02-02
- **Evaluador:** @governance

---

## RESUMEN EJECUTIVO

El documento institucional presenta técnicas ML de nivel profesional que **explican** por qué nuestro sistema actual tiene ~50% accuracy (D-024).

### Diagnóstico Principal
La causa raíz del problema D-024 (features no predictivos) es:

| Factor | Sistema Actual | Sistema Institucional |
|--------|---------------|----------------------|
| Target | Next-bar direction | **Triple Barrier** (TP/SL/Time) |
| CV | Standard K-Fold | **Purged K-Fold** (no leakage) |
| Sample Weight | Ninguno | **Uniqueness + Time Decay** |
| Features | 45 técnicos básicos | **+Microstructure +Entropy** |
| Meta-Labeling | No | **Sí** (side + size separados) |

---

## ANÁLISIS DETALLADO

### 1. TARGET ENGINEERING: Triple Barrier vs Next-Day Direction

**Problema actual:**
- Target: `y = sign(close[t+1] - close[t])` (sube/baja)
- Esto es esencialmente ruido para predicción
- ~50% accuracy = random

**Solución institucional: Triple Barrier**
```
Label = {
    +1: Take Profit hit first (long profitable)
    -1: Stop Loss hit first (short profitable)
     0: Timeout (neutral/no trade)
}
```

**Beneficios:**
- Refleja trading real (siempre tienes SL/TP)
- Considera el path, no solo el endpoint
- Labels más balanceados y significativos
- Filtra ruido (min_ret threshold)

**RECOMENDACIÓN: R-005**
Implementar Triple Barrier Labeling como target para ML y reward function de RL.

### 2. PURGED K-FOLD CROSS-VALIDATION

**Problema actual:**
- Standard K-Fold puede tener data leakage
- Labels que se solapan temporalmente contaminan train/test
- Resultados sobreestimados

**Solución institucional:**
```
┌────────┬────────┬────────┬────────┬────────┐
│ TRAIN  │EMBARGO │  TEST  │ TRAIN  │ TRAIN  │  Fold 1
├────────┼────────┼────────┼────────┼────────┤
│ TRAIN  │ TRAIN  │EMBARGO │  TEST  │ TRAIN  │  Fold 2
└────────┴────────┴────────┴────────┴────────┘

Embargo = gap temporal para prevenir leakage
Purge = eliminar samples con labels que se solapan con test
```

**RECOMENDACIÓN: R-006**
Implementar PurgedKFold para todas las validaciones ML.

### 3. SAMPLE WEIGHTING

**Problema actual:**
- Todos los samples tienen peso igual
- Samples con labels superpuestos sesgan el modelo
- No hay decaimiento temporal

**Solución institucional:**
```python
weight = uniqueness * time_decay

uniqueness = 1 / concurrent_labels  # Menos peso si muchos labels activos
time_decay = linspace(0.5, 1.0)     # Samples recientes valen más
```

**RECOMENDACIÓN: R-007**
Implementar sample weighting basado en uniqueness.

### 4. FEATURES INSTITUCIONALES

**Features actuales (45):**
- Returns: 5 períodos
- Volatilidad: ATR, BB
- Momentum: RSI, MACD, SMA ratios
- Volume: OBV, volumen relativo
- Precio: BB position, distancia a SMA

**Features institucionales adicionales:**

| Categoría | Features | Valor |
|-----------|----------|-------|
| **Microstructure** | Kyle Lambda, Amihud, VPIN, Roll Spread | Capturan liquidez e información |
| **Entropy** | Shannon, Approximate, Lempel-Ziv | Miden predictibilidad |
| **Fractional Diff** | Precios diferenciados d=0.3-0.5 | Estacionarios con memoria |
| **Cross-Asset** | Correlaciones, Beta rolling | Relaciones entre activos |
| **Volatility Adv** | Garman-Klass, Parkinson, VoV | Estimadores más eficientes |

**RECOMENDACIÓN: R-008**
Añadir features de microstructure y entropy al observation space.

### 5. META-LABELING

**Concepto:**
Separar la predicción en dos pasos:
1. **Primary Model**: Predice DIRECCIÓN (long/short)
2. **Secondary Model**: Predice si el primary ACIERTA (bet/no bet)

**Beneficio:**
- Aumenta precision (menos false positives)
- Permite bet sizing proporcional a confianza
- El 500-agent voting system ya hace algo similar

**RECOMENDACIÓN: R-009**
Evaluar meta-labeling para el voting system.

### 6. BET SIZING

**Sistema actual:**
- confidence_scaled con thresholds fijos (0.6, 0.7, 0.8, 0.9)
- Discretización en escalones

**Sistema institucional:**
- Kelly Criterion: `f* = (p*b - q) / b`
- Fraction conservador: 0.25 Kelly
- Ajuste dinámico por volatilidad

**RECOMENDACIÓN: R-010**
Implementar Kelly criterion fraccionado para position sizing.

---

## MATRIZ DE PRIORIZACIÓN

| Recomendación | Impacto | Esfuerzo | Prioridad | Dependencia |
|---------------|---------|----------|-----------|-------------|
| R-005: Triple Barrier | ALTO | MEDIO | 1 | - |
| R-006: Purged K-Fold | ALTO | BAJO | 2 | - |
| R-007: Sample Weights | MEDIO | BAJO | 3 | R-005 |
| R-008: Features Inst. | MEDIO | MEDIO | 4 | - |
| R-009: Meta-Labeling | MEDIO | ALTO | 5 | R-005, R-006 |
| R-010: Kelly Sizing | BAJO | BAJO | 6 | - |

---

## PLAN DE IMPLEMENTACIÓN PROPUESTO

### Sprint 1: Fundamentos (Semana actual)
1. **R-005**: Implementar `TripleBarrierLabeling` class
2. **R-006**: Implementar `PurgedKFold` class
3. Re-validar features con nuevo target

### Sprint 2: Features (Siguiente semana)
1. **R-007**: Añadir sample weighting
2. **R-008**: Implementar microstructure features
3. Re-entrenar modelos con features mejorados

### Sprint 3: Optimización (Semana 3)
1. **R-009**: Evaluar meta-labeling
2. **R-010**: Kelly criterion para voting system
3. Comparar resultados vs baseline actual

---

## CRITERIOS DE ACEPTACIÓN

| Métrica | Valor Actual | Objetivo Mínimo | Objetivo Bueno |
|---------|--------------|-----------------|----------------|
| CV Accuracy | ~50% | >52% | >55% |
| Sharpe Ratio | ~0.71 (test) | >1.0 | >1.5 |
| Max Drawdown | N/A | <15% | <10% |
| Win Rate | N/A | >50% | >55% |

---

## ARCHIVOS A CREAR

```
ml_institutional/
├── triple_barrier.py       # R-005
├── purged_kfold.py         # R-006
├── sample_weights.py       # R-007
├── microstructure.py       # R-008
├── entropy_features.py     # R-008
├── meta_labeling.py        # R-009
├── kelly_sizing.py         # R-010
└── institutional_system.py # Sistema completo
```

---

## RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Triple Barrier reduce samples | MEDIA | MEDIO | Ajustar min_ret threshold |
| Microstructure necesita tick data | MEDIA | BAJO | Usar proxies con OHLCV |
| Purged KFold reduce train size | BAJA | BAJO | Ajustar embargo % |
| Overfitting a nuevos features | MEDIA | ALTO | Validación estricta |

---

## DECISIÓN GOVERNANCE

**Veredicto:** ADOPTAR TÉCNICAS INSTITUCIONALES

**Justificación:**
1. El diagnóstico D-024 (50% accuracy) se explica por técnicas subóptimas
2. Las técnicas institucionales están probadas (Marcos López de Prado)
3. El esfuerzo de implementación es moderado
4. El potencial de mejora es alto

**Próximo paso:**
Implementar R-005 (Triple Barrier) y R-006 (Purged K-Fold) como primera iteración.

---

## REFERENCIAS

1. **Advances in Financial Machine Learning** - Marcos López de Prado
2. **Machine Learning for Asset Managers** - Marcos López de Prado
3. D-024: Finding - Features ~50% accuracy
4. Phase 4 results in autonomous_config.yaml

---

*Evaluación generada por @governance*
*Fecha: 2026-02-02*
