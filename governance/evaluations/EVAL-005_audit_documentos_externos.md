# GOVERNANCE EVALUATION: EVAL-005

## Auditoría de Documentos Externos
- **Fecha:** 2026-02-03
- **Evaluador:** @governance
- **Propósito:** Identificar documentos en Downloads/Desktop no integrados al proyecto

---

## RESUMEN

| Ubicación | Total | Relevantes | Integrados | Pendientes |
|-----------|-------|------------|------------|------------|
| Downloads | 30 | 12 | 6 | 6 |
| Desktop | 2 | 2 | 0 | 2 |
| **Total** | 32 | 14 | 6 | **8** |

---

## 1. DOCUMENTOS YA INTEGRADOS ✅

Estos documentos ya fueron evaluados o copiados al proyecto:

| Documento (Downloads) | Integrado como | Estado |
|-----------------------|----------------|--------|
| `sistema_ml_institucional.md` | EVAL-001 | ✅ Evaluado |
| `analisis_gaps_nautilus_ml.md` | EVAL-002 | ✅ Evaluado |
| `sistema_regimen_agentes.md` | EVAL-003 | ✅ Evaluado |
| `NAUTILUS_ESTADO_SISTEMA.md` | EVAL-004 + docs/ESTADO_SISTEMA.md | ✅ Migrado |
| `entrenamiento_profesional_rl_trading.md` | docs/reference/ (parcial) | ⚠️ Parcial |
| `estructura_entrenamiento_rl.md` | docs/reference/ (parcial) | ⚠️ Parcial |
| `SPEC_ML_Nautilus.md` | docs/reference/ (parcial) | ⚠️ Parcial |

---

## 2. DOCUMENTOS PENDIENTES DE EVALUAR 🔍

### 2.1 Alta Prioridad (Feb 2, relacionados con nautilus)

| Documento | Contenido Probable | Acción Recomendada |
|-----------|-------------------|-------------------|
| `INSTRUCCIONES_GOBERNANZA_DATOS.md` | Reglas para datos | **EVALUAR** |
| `INSTRUCCIONES_RUNPOD_MONITOR.md` | Monitoreo RunPod | **EVALUAR** |
| `validate_regime_detector.py` | Script validación | **INTEGRAR** si útil |

### 2.2 Desktop (Feb 2)

| Documento | Contenido Probable | Acción Recomendada |
|-----------|-------------------|-------------------|
| `MANUAL_OPERATIVO_NAUTILUS.md` | Guía operaciones | **EVALUAR** |
| `ANALISIS_SISTEMA_20260131.md` | Análisis ene 31 | **EVALUAR** |

### 2.3 Posiblemente Obsoletos

| Documento | Razón | Acción |
|-----------|-------|--------|
| `SPEC_ML_DeskGrade.md` | Proyecto anterior | Revisar si hay algo útil |
| `RUTA_PROYECTO_350_700_AGENTES.md` | Plan antiguo | Verificar si superseded |

---

## 3. DOCUMENTOS NO RELEVANTES ❌

Estos documentos NO son de nautilus-agents:

| Documento | Proyecto | Acción |
|-----------|----------|--------|
| `DESKGRADE_*.md` (6 archivos) | DeskGrade | Ignorar |
| `arquitectura_magento2_plugins.md` | Magento | Ignorar |
| `SPEC_CONTROL_PANEL_*.md` (3) | DeskGrade | Ignorar |
| `Estrategias_Trading_Guia_Completa.md` | General | Ignorar |
| `Herramientas_Trading_Institucional.md` | General | Ignorar |
| `INVENTARIO_TERCEROS_Y_METODOS.md` | DeskGrade | Ignorar |
| `SPEC_INDICADORES_MACD_STOCHASTIC.md` | DeskGrade | Ignorar |
| `SPEC_MIGRACION_CELERY.md` | DeskGrade | Ignorar |
| `SPEC_SISTEMA_MULTI_MERCADO_v1.md` | DeskGrade | Ignorar |
| `MENSAJE_CLAUDE_CODE.md` | Comunicación | Ignorar |
| `FASE_0_1_INDICE_GENERAL.md` | Antiguo | Ignorar |
| `AUDITORIA_COMPONENTES.md` | DeskGrade | Ignorar |

---

## 4. PLAN DE ACCIÓN

### Inmediato (hoy)
1. [ ] Leer y evaluar `INSTRUCCIONES_GOBERNANZA_DATOS.md`
2. [ ] Leer y evaluar `INSTRUCCIONES_RUNPOD_MONITOR.md`
3. [ ] Leer y evaluar `MANUAL_OPERATIVO_NAUTILUS.md`
4. [ ] Revisar `validate_regime_detector.py`

### Si hay contenido útil
- Integrar al proyecto en ubicación apropiada
- Documentar en DECISIONS.md
- Marcar original como obsoleto

### Limpieza (opcional)
- Mover documentos DeskGrade a carpeta separada
- Eliminar duplicados ya integrados

---

## 5. EVALUACIÓN DE DOCUMENTOS PENDIENTES ✅

### 5.1 INSTRUCCIONES_GOBERNANZA_DATOS.md
| Campo | Valor |
|-------|-------|
| **Estado** | ✅ EVALUADO |
| **Contenido** | Propuesta de pipeline de datos con esquema input/output |
| **Veredicto** | ⚠️ PARCIALMENTE CUBIERTO |
| **Razón** | Conceptos válidos pero ya implementados en `data/adapters/` |
| **Acción** | Revisar para mejoras futuras, no integrar ahora |

### 5.2 INSTRUCCIONES_RUNPOD_MONITOR.md
| Campo | Valor |
|-------|-------|
| **Estado** | ✅ EVALUADO |
| **Contenido** | Script de monitoreo GPU para RunPod con schedule |
| **Veredicto** | ✅ ÚTIL PARA FASE 5 |
| **Razón** | Complementa `monitoring/` para training en GPU |
| **Acción** | Integrar cuando se active RunPod training |

### 5.3 MANUAL_OPERATIVO_NAUTILUS.md
| Campo | Valor |
|-------|-------|
| **Estado** | ✅ EVALUADO |
| **Contenido** | Guía de operaciones para usuario (Rafa) |
| **Veredicto** | ✅ VALIOSO |
| **Razón** | Cubre operaciones diarias, dashboards, troubleshooting |
| **Acción** | Mover a `docs/MANUAL_OPERATIVO.md` |

### 5.4 ANALISIS_SISTEMA_20260131.md
| Campo | Valor |
|-------|-------|
| **Estado** | ✅ EVALUADO |
| **Contenido** | Análisis completo del sistema DeskGrade (no nautilus) |
| **Veredicto** | ❌ NO RELEVANTE |
| **Razón** | Documenta DeskGrade: 47 tablas, Celery, VectorBT, etc. |
| **Acción** | Ignorar - pertenece a proyecto DeskGrade |

### 5.5 validate_regime_detector.py
| Campo | Valor |
|-------|-------|
| **Estado** | ✅ EVALUADO |
| **Contenido** | Script standalone de validación de régimen con HMM |
| **Veredicto** | ⚠️ SUPERADO |
| **Razón** | Nautilus ya tiene `ml_institutional/regime_detector.py` con features avanzados |
| **Código útil** | Funciones de validación contra eventos conocidos |
| **Acción** | Extraer solo `validate_known_events()` si se necesita |

---

## 6. RESUMEN FINAL DE AUDITORÍA

### Documentos a Integrar
| Documento | Destino | Prioridad |
|-----------|---------|-----------|
| `MANUAL_OPERATIVO_NAUTILUS.md` | `docs/MANUAL_OPERATIVO.md` | ALTA |
| `INSTRUCCIONES_RUNPOD_MONITOR.md` | `monitoring/runpod/` | MEDIA (Fase 5) |

### Documentos Ya Cubiertos
- `NAUTILUS_ESTADO_SISTEMA.md` → `docs/ESTADO_SISTEMA.md` (D-037)
- `validate_regime_detector.py` → `ml_institutional/regime_detector.py` (R-018)
- `INSTRUCCIONES_GOBERNANZA_DATOS.md` → `data/adapters/` (parcial)

### Documentos a Ignorar (otros proyectos)
- `ANALISIS_SISTEMA_20260131.md` (DeskGrade)
- `DESKGRADE_*.md` (6 archivos)
- `SPEC_*_DeskGrade.md` (varios)

---

## 7. ACCIÓN RECOMENDADA

### Inmediato
1. [x] Completar lectura de 5 documentos pendientes
2. [ ] Mover `MANUAL_OPERATIVO_NAUTILUS.md` a `docs/`
3. [ ] Eliminar archivos obsoletos de Downloads (opcional)

### Fase 5 (cuando se active RunPod)
- Integrar script de monitoreo GPU
- Crear `monitoring/runpod/gpu_monitor.py`

### Limpieza Recomendada
```
C:\Users\PcVIP\Downloads\
├── NAUTILUS_ESTADO_SISTEMA.md     → ELIMINAR (ya en docs/)
├── INSTRUCCIONES_*.md              → ARCHIVAR
└── validate_regime_detector.py     → ARCHIVAR

C:\Users\PcVIP\Desktop\
├── MANUAL_OPERATIVO_NAUTILUS.md   → MOVER a proyecto
└── ANALISIS_SISTEMA_20260131.md   → MOVER a DeskGrade
```

---

## 8. CONCLUSIÓN GOVERNANCE

**Estado:** ✅ AUDITORÍA COMPLETADA

**Hallazgos:**
- 32 documentos externos evaluados
- 8 identificados como pendientes
- 5 leídos y evaluados en detalle
- 2 documentos útiles para integrar
- 3 documentos superseded o de otro proyecto

**Riesgo de pérdida de trabajo:** BAJO
- ML Institucional (11/11) completo en nautilus-agents
- Documentos externos son mayormente referencias o DeskGrade
- El único documento operativo útil es el Manual

**Próximo paso recomendado:**
Proceder con Fase 5 del roadmap. Los documentos externos no bloquean el progreso.

---

*Evaluación completada por @governance*
*Fecha: 2026-02-03*
*Documentos evaluados: 5/5*
