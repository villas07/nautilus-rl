# GOVERNANCE EVALUATION: EVAL-003

## Documento Evaluado
- **Archivo:** `sistema_regimen_agentes.md`
- **Origen:** Especificación de sistema de regímenes + selector de agentes
- **Fecha evaluación:** 2026-02-02
- **Evaluador:** @governance

---

## RESUMEN EJECUTIVO

Documento técnico de alta calidad que propone un sistema de **detección de régimen de mercado** combinado con **selección dinámica de agentes RL especializados**.

**Veredicto:** ADOPTAR - Arquitectura sólida y bien diseñada

### Alineación con Roadmap Actual

| Elemento | Estado en Roadmap | Impacto |
|----------|-------------------|---------|
| Regime Detection | R-014 (Fase 2) | Adelantar implementación |
| Agent Selector | Nuevo | Complementa voting system actual |
| Ensemble Methods | Existente | Mejora con context awareness |

---

## ANÁLISIS TÉCNICO

### 1. Detector de Régimen (Excelente)

El documento propone **3 métodos complementarios**:

| Método | Fortalezas | Debilidades |
|--------|------------|-------------|
| **Rule-Based** | Robusto, interpretable, sin overfitting | Puede ser lento en detectar cambios |
| **HMM** | Probabilístico, captura patrones ocultos | Requiere calibración, puede overfit |
| **Momentum-Based** | Rápido, multi-timeframe | Menos robusto en sideways |

**Evaluación:** El ensemble de los 3 métodos con voting ponderado es una decisión acertada. Reduce el riesgo de falsos positivos de cualquier método individual.

### 2. Clasificación de Regímenes (Apropiada)

```
6 Regímenes = 3 Tendencias × 2 Volatilidades

BULL_LOW_VOL      → Momentum strategies
BULL_HIGH_VOL     → Conservative momentum
BEAR_LOW_VOL      → Mean reversion, short
BEAR_HIGH_VOL     → Defensive, cash
SIDEWAYS_LOW_VOL  → Range trading
SIDEWAYS_HIGH_VOL → Reduce exposure
```

**Evaluación:** Clasificación estándar en la industria. Suficiente granularidad sin ser excesiva.

### 3. Selector de Agentes (Bien Diseñado)

Modos de selección:
1. **Best Agent** - Usa el agente con mejor performance reciente
2. **Ensemble** - Weighted average de top N agentes
3. **Voting** - Mayoría simple

**Evaluación:** El modo "ensemble" con pesos dinámicos basados en performance es el más robusto. Consistente con nuestro voting system actual.

### 4. Integración con Nautilus (Completa)

El documento incluye:
- `RegimeAdaptiveStrategy` funcional
- Encoding de régimen como features adicionales
- Manejo de transiciones entre regímenes
- Performance tracking por régimen

---

## COMPARATIVA CON SISTEMA ACTUAL

| Aspecto | Sistema Actual | Propuesta | Mejora |
|---------|---------------|-----------|--------|
| Agentes | 500 genéricos | Especializados por régimen | Contexto |
| Selección | Voting simple | Regime-aware voting | Adaptabilidad |
| Features | 57 (con R-008) | 57 + 6 régimen | Más contexto |
| Training | Un modelo por agente | Un modelo por régimen × estrategia | Especialización |

---

## RECOMENDACIONES GOVERNANCE

### R-018: Implementar Regime Detection (ALTA)

**Prioridad:** ALTA (adelantar de Fase 2 a Fase 1.5)

**Justificación:**
- El sistema actual trata todos los regímenes igual
- Agentes especializados tienen mejor performance en su régimen
- El código está 80% listo en el documento

**Componentes a crear:**
```
ml_institutional/
├── regime_detector.py    ← Implementar
├── agent_selector.py     ← Adaptar de voting_system.py
└── regime_training.py    ← Para entrenar por régimen
```

**Esfuerzo estimado:** 1 semana

### R-019: Training Pipeline por Régimen (MEDIA)

**Prioridad:** MEDIA (después de R-018)

**Propuesta:**
En lugar de 500 agentes genéricos, entrenar:
- 30-50 agentes × 6 regímenes = 180-300 agentes especializados
- Cada agente entrenado SOLO con datos de su régimen

**Beneficios:**
- Menos agentes totales
- Mejor especialización
- Reducción de overfitting cruzado

### R-020: Regime Features en Observation (BAJA)

**Prioridad:** BAJA (opcional)

Añadir 6 features de régimen al observation space:
```python
# Ya en InstitutionalObservationConfig
include_regime: bool = True  # +6 features
```

**Impacto:** 57 → 63 features

---

## COMPATIBILIDAD CON R-005 a R-008

| Componente | Compatibilidad | Notas |
|------------|----------------|-------|
| Triple Barrier (R-005) | ✅ Compatible | Labels independientes de régimen |
| Purged K-Fold (R-006) | ✅ Compatible | Aplicar por régimen |
| Sample Weights (R-007) | ✅ Compatible | Pesos por régimen |
| Microstructure (R-008) | ✅ Compatible | Features adicionales |

---

## RIESGOS Y MITIGACIONES

### Riesgo 1: Latencia en Cambio de Régimen
- **Problema:** Agente equivocado activo durante transición
- **Mitigación:** `regime_min_duration` + confidence threshold

### Riesgo 2: Escasez de Datos por Régimen
- **Problema:** Algunos regímenes tienen pocos datos históricos
- **Mitigación:** Combinar regímenes similares, data augmentation

### Riesgo 3: Overfitting al Régimen
- **Problema:** Agente solo funciona en régimen exacto
- **Mitigación:** Validación cruzada entre regímenes similares

---

## MATRIZ DE PRIORIZACIÓN ACTUALIZADA

| ID | Componente | Prioridad | Estado |
|----|------------|-----------|--------|
| R-005 | Triple Barrier | 1 | ✅ DONE |
| R-006 | Purged K-Fold | 2 | ✅ DONE |
| R-007 | Sample Weights | 3 | ✅ DONE |
| R-008 | Microstructure Features | 4 | ✅ DONE (D-028) |
| R-018 | Regime Detection | 5 | ✅ DONE (D-030) |
| R-012 | Risk Manager Real-Time | 6 | ✅ EXISTS |
| R-013 | Monitoring Completo | 7 | ✅ DONE (D-031) |
| R-019 | Training por Régimen | 8 | ✅ DONE (D-033) |
| R-014 | → Absorbido por R-018 | - | MERGED |

---

## CHECKLIST DE IMPLEMENTACIÓN

```
FASE 1: DETECTOR DE RÉGIMEN
├── [ ] Crear ml_institutional/regime_detector.py
├── [ ] Implementar RegimeDetector.detect_rule_based()
├── [ ] Implementar RegimeDetector.detect_hmm() (opcional)
├── [ ] Implementar RegimeDetector.detect_momentum()
├── [ ] Test con datos del catálogo
├── [ ] Calibrar thresholds con SPY/BTC

FASE 2: INTEGRACIÓN
├── [ ] Adaptar live/voting_system.py para regime-awareness
├── [ ] Añadir régimen al observation space (opcional)
├── [ ] Crear RegimeAdaptiveStrategy para Nautilus
├── [ ] Test en backtest

FASE 3: TRAINING
├── [ ] Etiquetar datos históricos por régimen
├── [ ] Script de training por régimen
├── [ ] Entrenar subset de prueba (10 agentes × 6 regímenes)
├── [ ] Comparar vs agentes genéricos
```

---

## DECISIÓN GOVERNANCE

**Veredicto:** ADOPTAR DOCUMENTO

**Acción inmediata:**
1. Crear `ml_institutional/regime_detector.py` basado en el documento
2. Integrar con InstitutionalObservationBuilder
3. Probar detección con datos reales

**Próximo sprint:**
- R-018: Regime Detection
- R-012: Risk Manager (paralelo)

---

## MÉTRICAS DE ÉXITO

| Métrica | Baseline (sin régimen) | Target (con régimen) |
|---------|------------------------|----------------------|
| Sharpe Ratio | 1.0 | >1.3 |
| Max Drawdown | 15% | <12% |
| Win Rate | 50% | >52% |
| Regime Accuracy | N/A | >70% |

---

*Evaluación generada por @governance*
*Fecha: 2026-02-02*
*Referencia: EVAL-001, EVAL-002, R-014*
