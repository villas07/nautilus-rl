# NAUTILUS AGENTS - ESTADO DEL SISTEMA

> Documento de referencia rápida para nuevas sesiones de Claude Code.
> **Última actualización:** 2026-02-03

---

## Quick Reference

| Aspecto | Valor |
|---------|-------|
| **Fase actual** | 5 (Escalar 500 Agentes) - Pendiente |
| **Fases completadas** | 1, 2, 3, 4 |
| **ML Institucional** | 11/11 recomendaciones ✅ |
| **Última decisión** | D-038 |
| **Training activo** | 6 agentes RunPod (batch_2) |
| **500 configs** | Generados en configs/agents_generated/ |
| **Regla catálogo** | Linux only (IMMUTABLE 3.3) |

---

## 1. ARQUITECTURA DE 3 NODOS

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PC WINDOWS    │     │  VPS HETZNER    │     │     RUNPOD      │
│                 │     │                 │     │                 │
│  • Desarrollo   │     │  • Datos raw    │     │  • GPU A100     │
│  • Código       │     │  • Catálogo     │     │  • Training     │
│  • Git          │     │  • Web server   │     │  • Temporal     │
│                 │     │                 │     │                 │
│  IP: local      │     │  46.225.11.110  │     │  (IP variable)  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │      SSH sin pwd      │     HTTP download     │
         │◄─────────────────────►│◄─────────────────────►│
         │                       │                       │
         │         git push      │   raw_data.tar.gz     │
         └───────────────────────┴───────────────────────┘
```

---

## 2. CONEXIONES SSH

### PC → VPS Hetzner
```bash
ssh root@46.225.11.110
# Funciona SIN password (clave configurada)
```

### PC → RunPod
```bash
# El ID cambia cada vez que se crea un pod
ssh [POD_ID]@ssh.runpod.io -i ~/.ssh/id_ed25519
# Clave pública configurada en RunPod Settings
```

### Archivos en VPS (http://46.225.11.110:8080/nautilus/)
- `raw_data.tar.gz` - Datos CSV (4.4MB)
- `code.tar.gz` - Código Python
- `pod_startup.sh` - Script de inicio RunPod

---

## 3. REGLA INMUTABLE: CATÁLOGO LINUX (D-019)

```
⚠️  INMUTABLE (Regla 3.3)

El catálogo NautilusTrader DEBE crearse en Linux, NUNCA en Windows.

Motivo: Parquet tiene problemas de serialización entre Windows y Linux.

Flujo correcto:
  CSV (cualquier lugar) → Catálogo (Linux) → Training (RunPod)

Script: scripts/create_catalog_linux.py
```

---

## 4. ESTADO DE COMPONENTES

### ML Institucional (100% completado)

| ID | Componente | Estado | Decisión |
|----|------------|--------|----------|
| R-005 | Triple Barrier Labeling | ✅ | D-025 |
| R-006 | Purged K-Fold CV | ✅ | D-025 |
| R-007 | Sample Weighting | ✅ | D-025 |
| R-008 | Microstructure/Entropy | ✅ | D-028 |
| R-009 | Meta-Labeling | ✅ | D-034 |
| R-010 | Kelly Criterion | ✅ | D-035 |
| R-011 | Data Pipeline Robustness | ✅ | D-032 |
| R-012 | Risk Manager | ✅ | EXISTS |
| R-013 | Monitoring | ✅ | D-031 |
| R-018 | Regime Detection | ✅ | D-030 |
| R-019 | Regime Training | ✅ | D-033 |

### Fases del Roadmap

| Fase | Nombre | Estado |
|------|--------|--------|
| 1 | Setup Inicial | ✅ Completada |
| 2 | Integración SB3→Nautilus | ✅ Completada |
| 3 | Baseline Estrategias | ✅ Completada |
| 4 | ML Supervisado | ✅ Completada |
| 5 | Escalar 500 Agentes | ⏳ Pendiente |
| 6 | Paper Trading | ⏳ Pendiente |
| 7 | Live Trading | ⏳ Pendiente |

---

## 5. RUTAS IMPORTANTES

### PC Windows
```
C:\Users\PcVIP\nautilus-agents\          # Código fuente
C:\Users\PcVIP\nautilus-agents\models\   # Modelos entrenados
C:\Users\PcVIP\.ssh\id_ed25519           # Clave SSH
```

### VPS Hetzner
```
/var/www/html/nautilus/                  # Archivos para RunPod
```

### RunPod
```
/workspace/nautilus-agents/              # Código
/workspace/data/catalog/                 # Catálogo Nautilus
/workspace/models/                       # Modelos guardados
```

---

## 6. CHECKLIST PRE-TRAINING

```
[ ] Catálogo creado en LINUX (nunca Windows)
[ ] Verificar: instruments > 0
[ ] Verificar: bars > 1000
[ ] Test rápido: episode_length > 100
[ ] Archivos en VPS actualizados:
    [ ] raw_data.tar.gz
    [ ] code.tar.gz
    [ ] pod_startup.sh
```

### Verificar catálogo
```python
from nautilus_trader.persistence.catalog import ParquetDataCatalog
catalog = ParquetDataCatalog('/workspace/data/catalog')
print(f'Instruments: {len(catalog.instruments())}')
print(f'Bars: {len(list(catalog.bars()))}')
```

---

## 7. DEBUGGING RÁPIDO

### Si episode_length = 1
- Catálogo creado en Windows (violación regla 3.3)
- Solución: Recrear en Linux

### Si reward = 0 siempre
- Verificar RewardCalculator en `gym_env/rewards.py`
- Verificar variación en precios

### Si SSH Permission denied
- RunPod: Verificar clave en Settings → SSH public keys
- VPS: Verificar `~/.ssh/authorized_keys`

---

## 8. DOCUMENTOS RELACIONADOS

| Documento | Propósito |
|-----------|-----------|
| `.roles/DECISIONS.md` | Historial de decisiones |
| `config/autonomous_config.yaml` | Configuración y roadmap |
| `.rules/IMMUTABLE_RULES.md` | Reglas permanentes |
| `CLAUDE.md` | Contexto para Claude Code |
| `docs/MANUAL_OPERATIVO.md` | Guía de comandos para el propietario |
| `governance/evaluations/` | Evaluaciones de governance |

---

## 9. PRÓXIMOS PASOS (Fase 5)

| Tarea | Descripción |
|-------|-------------|
| 5.1 | Crear 500 configs de hyperparámetros |
| 5.2 | Batch training en RunPod (~44h, ~$88) |
| 5.3 | Validación 5 filtros |
| 5.4 | Configurar ensemble/voting |

---

## 10. CONTACTO

- **Usuario:** Rafa
- **Experiencia:** PHP developer, 15-20 años
- **Horizonte inversión:** 15-20 años

---

*Documento gestionado por @governance*
*Referencia: EVAL-004*
*Versión: 1.0.0*
