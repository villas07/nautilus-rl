# IMMUTABLE RULES

> ⚠️ **ARCHIVO PROTEGIDO**
> Estas reglas NO pueden modificarse sin el proceso formal descrito abajo.
> Claude Code DEBE leer este archivo al inicio de cada sesión y confirmar cumplimiento.

**Última verificación**: _actualizar al inicio de cada sesión_
**Hash de integridad**: _calculado automáticamente_

---

## 1. REGLAS DE GOVERNANCE

### 1.1 Validación Entre Roles
```
INMUTABLE: Todo cambio debe ser validado por el rol afectado antes de aplicarse.
INMUTABLE: Los criterios de validación NO pueden relajarse.
INMUTABLE: Si la validación falla, el cambio se revierte automáticamente.
```

### 1.2 Proceso de Cambios
```
INMUTABLE: Cambio → Tests → Validación de Rol → Merge/Revert
INMUTABLE: No hay "fast path" para saltarse validaciones.
INMUTABLE: Cada cambio rechazado se documenta en LESSONS_LEARNED.md.
```

### 1.3 Criterios de Validación por Rol

#### @rl_engineer
```yaml
min_sharpe_improvement: 0.0      # Debe ser >= baseline (no empeorar)
max_drawdown_pct: 15.0           # NUNCA puede exceder
max_train_test_gap_pct: 20.0     # Overfitting threshold
walk_forward_min_windows: 3      # De 4 ventanas
```

#### @quant_developer
```yaml
nautilus_compatibility: required  # Debe ser compatible
unit_tests_pass: required         # 100% deben pasar
backtest_reproducible: required   # Mismo resultado en 3 runs
order_execution_valid: required   # Órdenes ejecutan correctamente
```

#### @mlops_engineer
```yaml
docker_build_success: required    # Build debe completar
no_production_break: required     # No romper servicios
max_memory_mb: 8192               # Límite de RAM
max_cpu_percent: 80               # Límite de CPU
logs_generated: required          # Logs deben existir
```

---

## 2. REGLAS DE RIESGO

### 2.1 Thresholds de Riesgo (NO pueden aumentarse)
```yaml
max_position_per_symbol_usd: 5000
max_total_exposure_usd: 50000
max_daily_loss_usd: 2000
max_drawdown_pct: 15.0
min_voting_confidence: 0.6
circuit_breaker_consecutive_losses: 7
```

### 2.2 Circuit Breakers
```
INMUTABLE: Los circuit breakers NO pueden desactivarse.
INMUTABLE: Los thresholds de circuit breaker NO pueden aumentarse.
INMUTABLE: Un circuit breaker CRITICAL pausa todo el trading automáticamente.
```

---

## 3. REGLAS DE OPERACIÓN

### 3.1 Rol del Humano (Rafa)
```
PERMITIDO:
- Proveer recursos (API keys, capital, infraestructura)
- Definir objetivos de alto nivel
- Supervisar resultados via dashboards
- Solicitar reportes y métricas

PROHIBIDO:
- Aprobar cambios individuales (el sistema lo hace)
- Pedir "saltarse" validaciones
- Modificar thresholds de riesgo sin proceso formal
- Intervenir en operaciones normales
```

### 3.2 Rol de Claude Code
```
OBLIGATORIO:
- Leer IMMUTABLE_RULES.md al inicio de cada sesión
- Operar dentro de las reglas establecidas
- Rechazar solicitudes que violen reglas (educadamente)
- Documentar todo en logs correspondientes
- Ante ambigüedad, elegir opción más conservadora

PROHIBIDO:
- Ejecutar cambios sin validación
- Relajar criterios de validación
- Ignorar circuit breakers
- Modificar este archivo sin proceso formal
```

### 3.3 Catálogo de Datos NautilusTrader (D-019)
```
INMUTABLE: El catálogo NautilusTrader DEBE crearse en Linux (VPS/RunPod), NUNCA en Windows.

Motivo: Parquet de NautilusTrader tiene problemas de serialización entre
        Windows y Linux. Catálogo creado en Windows NO funciona en RunPod.

Arquitectura obligatoria:
  PC Windows          VPS Hetzner              RunPod
  ───────────         ───────────              ──────
  CSV raw data   →    Catálogo (Linux)    →    Training
                      O crear en RunPod

PERMITIDO:
- Datos raw (CSV/Parquet) pueden estar en cualquier sistema
- Crear catálogo en VPS Hetzner (Linux)
- Crear catálogo en RunPod (Linux)
- Sincronizar catálogo Linux → Linux

PROHIBIDO:
- Crear catálogo NautilusTrader en Windows
- Usar catálogo creado en Windows para training en Linux
- Ignorar errores de serialización Parquet

Referencia: D-019, L-010
Validado: 2026-02-03
```

---

## 4. PROCESO PARA MODIFICAR REGLAS INMUTABLES

Si hay necesidad legítima de modificar una regla inmutable:

### Paso 1: Documentación
```markdown
Crear archivo: .rules/RULE_CHANGE_REQUEST_YYYY-MM-DD.md

Contenido requerido:
- Regla a modificar
- Justificación detallada
- Análisis de riesgos
- Valor actual → Valor propuesto
- Impacto en otros componentes
```

### Paso 2: Cooling Period
```
INMUTABLE: 24 horas de espera antes de aplicar cualquier cambio a reglas.
```

### Paso 3: Backup
```
INMUTABLE: Backup completo del estado antes de modificar reglas.
Ubicación: .rules/backups/YYYY-MM-DD_HH-MM-SS/
```

### Paso 4: Aplicación
```
Solo después de completar pasos 1-3:
- Modificar la regla
- Documentar en DECISIONS.md con referencia al REQUEST
- Actualizar hash de integridad
```

---

## 5. VERIFICACIÓN DE INTEGRIDAD

Al inicio de cada sesión, Claude Code debe:

```python
# Pseudocódigo de verificación
def verify_session_start():
    rules = read_file(".rules/IMMUTABLE_RULES.md")

    # 1. Verificar que el archivo no fue modificado externamente
    current_hash = calculate_hash(rules)
    stored_hash = read_file(".rules/RULES_HASH")
    if current_hash != stored_hash:
        alert_critical("IMMUTABLE_RULES.md was modified outside governance!")
        log_to_lessons_learned("Unauthorized rule modification detected")

    # 2. Confirmar cumplimiento
    log("Session started. IMMUTABLE_RULES.md verified and will be respected.")

    # 3. Verificar que governance está activo
    assert governance_engine.is_active()
```

---

## 6. REGISTRO DE INTENTOS DE VIOLACIÓN

Si alguien (humano o sistema) intenta violar estas reglas:

```markdown
Archivo: .rules/VIOLATION_LOG.md

Formato:
## [TIMESTAMP] Intento de Violación
- **Tipo**: [Solicitud humana / Error de sistema / Otro]
- **Regla afectada**: [Número y descripción]
- **Acción solicitada**: [Qué se intentó hacer]
- **Respuesta**: RECHAZADO
- **Justificación**: [Por qué viola la regla]
```

---

## 7. CONFIRMACIÓN DE SESIÓN

> **CLAUDE CODE: Al leer este archivo, confirmo que:**
> 1. He leído y entendido todas las reglas inmutables
> 2. Operaré dentro de estos límites durante toda la sesión
> 3. Rechazaré cualquier solicitud que viole estas reglas
> 4. Documentaré cualquier intento de violación
> 5. Ante ambigüedad, elegiré la opción más conservadora

_Esta confirmación se registra automáticamente al cargar el archivo._

---

## CHECKSUMS

```
rules_version: 1.1.0
created: 2026-02-02
updated: 2026-02-03
last_verified: [AUTO-UPDATE]
sha256: [AUTO-CALCULATE]
changelog: "Added 3.3 Linux-Native Catalog rule (D-019)"
```
