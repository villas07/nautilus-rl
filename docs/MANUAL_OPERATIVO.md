# MANUAL OPERATIVO NAUTILUS-AGENTS
## Guía de Comandos y Operaciones para el Propietario del Sistema

> **Migrado desde:** `C:\Users\PcVIP\Desktop\MANUAL_OPERATIVO_NAUTILUS.md`
> **Fecha migración:** 2026-02-03
> **Referencia:** EVAL-005

---

## TU ROL: PROPIETARIO DEL LABORATORIO

**Lo que haces:**
- Provees recursos (API keys, capital, infraestructura)
- Defines objetivos de alto nivel
- Supervisas resultados en dashboard
- Recibes alertas críticas por Telegram

**Lo que NO haces:**
- Tomar decisiones operativas
- Aprobar cambios individuales
- Elegir entre opciones técnicas
- Definir parámetros específicos

---

## COMANDOS RÁPIDOS POR SITUACIÓN

### INICIO DE SESIÓN
```
¿Cómo está el sistema?
```
Respuesta esperada: Estado completo con P&L, agentes activos, alertas.

Si no ejecuta protocolo:
```
Ejecuta el protocolo de inicio primero
```

---

### CUANDO CODE PREGUNTA OPCIONES

**Code:** "¿Cuál prefieres? 1, 2 o 3"

**Tú:**
```
gobernanza
```
o
```
Aplica governance autónomo. Los roles deciden según criterios profesionales.
```

---

### DECISIONES PENDIENTES (DQ-XXX)

**Ver decisiones pendientes:**
```
¿Qué decisiones están pendientes?
```

**Resolver decisión con governance:**
```
@team_review DQ-001 - Los roles deben decidir según estándares de la industria.
```

**Forzar resolución autónoma:**
```
Resolved DQ-XXX con governance autónomo. No me consultes, aplicad criterios profesionales.
```

---

### INVOCAR ROLES ESPECÍFICOS

| Comando | Cuándo usar |
|---------|-------------|
| `@rl_engineer [tarea]` | Entrenamiento, validación, modelos |
| `@quant_developer [tarea]` | Trading, backtesting, estrategias |
| `@mlops_engineer [tarea]` | Docker, deployment, infraestructura |
| `@team_review DQ-XXX` | Decisiones que requieren múltiples roles |
| `@retrospective` | Revisar qué salió bien/mal |

---

### DATOS Y FUENTES

**Añadir nueva fuente de datos:**
```
Añade [FUENTE] como fuente de datos. API key: [KEY]
```
El sistema evaluará automáticamente con governance.

**Descargar datos de una región:**
```
Descarga datos históricos de [mercados/región]
```

**Verificar catálogo:**
```
¿Qué datos tenemos en el catálogo?
```

---

### ENTRENAMIENTO

**Entrenar agentes (local):**
```
Entrena agentes con los datos actuales
```
Governance decidirá cantidad y configuración.

**Preparar entrenamiento GPU remoto:**
```
@mlops_engineer Prepara el sistema para entrenar 500 agentes en GPU remota (RunPod/Vast.ai)
```

**Ver estado de entrenamiento:**
```
¿Cómo va el entrenamiento?
```

**Detener entrenamiento:**
```
Para el entrenamiento actual
```

---

### VALIDACIÓN DE MODELOS

**Validar modelos entrenados:**
```
@rl_engineer Valida los modelos entrenados con los 5 filtros
```

**Ver resultados de validación:**
```
¿Qué modelos pasaron validación?
```

**Definir thresholds de validación:**
```
@team_review DQ-001 - Definid thresholds profesionales basados en estándares de la industria
```

---

### LIVE TRADING

**Activar paper trading:**
```
Activa paper trading con los modelos validados
```

**Ver posiciones actuales:**
```
¿Qué posiciones tenemos abiertas?
```

**Pausar todo:**
```
Pausa todos los agentes
```

**Reactivar:**
```
Reactiva los agentes pausados
```

---

### SITUACIONES DE EMERGENCIA

**Sistema no responde como debe:**
```
Lee CLAUDE.md y .rules/IMMUTABLE_RULES.md. Confirma que entiendes las reglas antes de continuar.
```

**Code se salta governance:**
```
Acabas de saltarte el governance. Revierte el cambio y hazlo correctamente.
```

**Code pide aprobación operativa:**
```
No me pidas aprobación operativa. El sistema es autónomo. Governance decide.
```

**Algo se rompió:**
```
@retrospective ¿Qué falló y por qué? Documenta en LESSONS_LEARNED.md
```

---

### MANTENIMIENTO

**Verificar integridad del sistema:**
```
Verifica integridad del sistema
```

**Ver logs de decisiones:**
```
Muéstrame las últimas decisiones en DECISIONS.md
```

**Ver errores pasados:**
```
Muéstrame LESSONS_LEARNED.md
```

**Backup del sistema:**
```
@mlops_engineer Haz backup completo del sistema
```

---

## THRESHOLDS ESTÁNDAR DE LA INDUSTRIA

Referencia para cuando governance necesite decidir:

### Validación de Modelos RL (Trading)

| Métrica | Mínimo Aceptable | Bueno | Excelente |
|---------|------------------|-------|-----------|
| Sharpe Ratio | > 1.0 | > 1.5 | > 2.0 |
| Sortino Ratio | > 1.2 | > 1.8 | > 2.5 |
| Max Drawdown | < 20% | < 15% | < 10% |
| Win Rate | > 45% | > 52% | > 58% |
| Profit Factor | > 1.2 | > 1.5 | > 2.0 |
| Calmar Ratio | > 0.5 | > 1.0 | > 2.0 |

### Validación Anti-Overfitting

| Criterio | Límite |
|----------|--------|
| Train vs Test gap | < 20% |
| Walk-forward efficiency | > 50% |
| Out-of-sample Sharpe | > 70% del in-sample |
| Cross-validation std | < 30% de la media |

### Risk Management (Live)

| Parámetro | Valor Conservador | Moderado | Agresivo |
|-----------|-------------------|----------|----------|
| Max position size | 2% capital | 5% capital | 10% capital |
| Max daily loss | 1% capital | 2% capital | 3% capital |
| Max correlation entre agentes | 0.3 | 0.5 | 0.7 |
| Min agentes activos | 5 | 10 | 3 |

---

## ARQUITECTURA DEL SISTEMA

```
┌─────────────────────────────────────────────────────────────┐
│                    TÚ (Propietario)                         │
│         Dashboard Grafana + Alertas Telegram                │
└─────────────────────────┬───────────────────────────────────┘
                          │ Solo supervisión
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  GOVERNANCE AUTÓNOMO                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │@rl_engineer │ │@quant_dev   │ │@mlops_eng   │           │
│  │             │ │             │ │             │           │
│  │- Modelos    │ │- Trading    │ │- Infra      │           │
│  │- Validación │ │- Backtest   │ │- Docker     │           │
│  │- Rewards    │ │- Ejecución  │ │- Deploy     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                          │                                  │
│              IMMUTABLE_RULES.md                             │
│              (No se puede saltar)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 NAUTILUS TRADER                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Backtest │ │   Live   │ │   Data   │ │  Orders  │       │
│  │  Engine  │ │  Engine  │ │ Catalog  │ │  Engine  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    BROKERS                                  │
│     Interactive Brokers (Stocks, Futures, Forex)            │
│     Binance (Crypto)                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## ESTRUCTURA DE ARCHIVOS CLAVE

| Archivo | Propósito | ¿Puedes modificar? |
|---------|-----------|-------------------|
| `CLAUDE.md` | Instrucciones para Claude Code | No (solo con proceso formal) |
| `.rules/IMMUTABLE_RULES.md` | Reglas que no se pueden saltar | No (requiere 24h cooling) |
| `.roles/DECISION_QUEUE.md` | Decisiones pendientes | Solo lectura |
| `.roles/DECISIONS.md` | Historial de decisiones | Solo lectura |
| `.roles/LESSONS_LEARNED.md` | Errores y aprendizajes | Solo lectura |
| `config/autonomous_config.yaml` | Configuración autónoma | No (solo via governance) |
| `.env` | API keys y secretos | Sí (puedes añadir keys) |

---

## CHECKLIST DIARIO (5 minutos)

1. [ ] Abrir Grafana -> Ver P&L y estado
2. [ ] Revisar Telegram -> ¿Alertas importantes?
3. [ ] Pregunta rápida: `¿Cómo está el sistema?`
4. [ ] Si hay decisiones pendientes: `@team_review DQ-XXX`
5. [ ] Cerrar y dejar que opere

---

## CHECKLIST SEMANAL (30 minutos)

1. [ ] `@retrospective` - Revisar semana
2. [ ] Ver `LESSONS_LEARNED.md`
3. [ ] Ver `DECISIONS.md` - ¿Decisiones correctas?
4. [ ] Comparar P&L real vs esperado
5. [ ] `@mlops_engineer` - ¿Backup necesario?

---

## CONTACTOS Y RECURSOS

- **NautilusTrader Discord:** Soporte técnico
- **Documentación:** https://nautilustrader.io/docs
- **Tu cuenta IB:** DUO275624 (paper trading)
- **Telegram Bot:** Configurado para alertas

---

## FRASES CLAVE PARA RECORDAR

> "Gobernanza" - Cuando Code pide opciones
>
> "@team_review" - Para decisiones importantes
>
> "Los roles deciden" - No tú
>
> "Estándares de la industria" - No inventes thresholds
>
> "Sistema autónomo" - No pidas aprobación

---

*Documento creado: 2026-02-02*
*Migrado al proyecto: 2026-02-03*
*Sistema: Nautilus-Agents v1.0*
*Propietario: Rafa*
