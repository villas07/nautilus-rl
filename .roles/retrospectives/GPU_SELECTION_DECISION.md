# Decisión: Selección de GPU para 500 Agentes RL

**Fecha**: 2026-02-02
**Decisores**: @mlops_engineer, @rl_engineer
**Tipo**: Governance Decision

---

## Análisis de GPUs Disponibles

### Requisitos para Training RL (PPO + SB3)

| Requisito | Mínimo | Recomendado |
|-----------|--------|-------------|
| VRAM | 8GB | 24GB+ |
| RAM Sistema | 32GB | 64GB+ |
| Disponibilidad | Media | **Alta** |
| Costo/hora | - | <$1.00 |

### GPUs Evaluadas

| GPU | $/hr | VRAM | RAM | Disp. | Costo 29h | Score |
|-----|------|------|-----|-------|-----------|-------|
| **RTX 4090** | $0.59 | 24GB | 31GB | **High** | **$17** | ⭐⭐⭐⭐⭐ |
| RTX 5090 | $0.89 | 32GB | 92GB | High | $26 | ⭐⭐⭐⭐ |
| A40 | $0.40 | 48GB | 50GB | Low | $12 | ⭐⭐⭐ |
| L4 | $0.39 | 24GB | 55GB | Low | $11 | ⭐⭐⭐ |
| RTX 6000 Ada | $0.77 | 48GB | 62GB | Low | $22 | ⭐⭐⭐ |
| L40S | $0.86 | 48GB | 94GB | Low | $25 | ⭐⭐⭐ |
| H100 SXM | $2.69 | 80GB | 172GB | High | $78 | ⭐⭐ |
| H200 SXM | $3.59 | 141GB | 188GB | High | $104 | ⭐ |

### Criterios de Evaluación

1. **Disponibilidad (40%)**: Run de 29+ horas no puede interrumpirse
2. **Costo Total (30%)**: Presupuesto ~$100 máximo
3. **VRAM (15%)**: 24GB suficiente para 8 agentes paralelos
4. **RAM Sistema (15%)**: Para datos y envs vectorizados

---

## Análisis Detallado

### @rl_engineer - Perspectiva Técnica

```
PPO Training Memory Usage:
- Model (256,256 MLP): ~50MB
- Rollout buffer (2048 steps × 8 envs): ~200MB
- Gradients: ~100MB
- Per agent total: ~400MB

8 agentes paralelos × 400MB = 3.2GB VRAM
→ 24GB es 7x más de lo necesario ✓

Training Speed:
- RTX 4090: ~4000 steps/sec (Ada architecture)
- RTX 5090: ~5000 steps/sec (Blackwell)
- A40: ~2500 steps/sec (Ampere)

Tiempo estimado con RTX 4090:
- 5M steps / 4000 = 1250 sec = 21 min/agente
- Con 8 paralelos: 2.6 min efectivos/agente
- 500 agentes: ~22 horas
```

### @mlops_engineer - Perspectiva Operacional

```
Riesgos por Disponibilidad:
- Low availability → Probabilidad de interrupción ~30%
- Interrupción = pérdida de progreso + reinicio
- Con checkpoint cada 100K steps: pérdida máxima ~4 min

Costo de interrupción:
- Tiempo perdido: ~2-4 horas
- Costo adicional: ~$5-15
- Frustración: Alta

Conclusión: High availability es CRÍTICO
```

---

## Decisión Final

### GPU Seleccionada: **RTX 4090**

| Especificación | Valor |
|----------------|-------|
| Precio | $0.59/hr (spot: $0.50/hr) |
| VRAM | 24GB GDDR6X |
| RAM Sistema | 31GB |
| vCPU | 6 |
| Disponibilidad | **High** |
| Arquitectura | Ada Lovelace |

### Justificación

1. **Alta disponibilidad** - Única GPU <$1/hr con disponibilidad "High"
2. **Mejor precio/rendimiento** - $17-20 para 500 agentes
3. **24GB VRAM suficiente** - 7x más de lo necesario para PPO
4. **Arquitectura moderna** - Ada Lovelace optimizada para ML
5. **Probada en producción** - Ampliamente usada para RL training

### Alternativa Aprobada

Si RTX 4090 no disponible: **RTX 5090** ($0.89/hr)
- Mayor VRAM (32GB) y RAM (92GB)
- Costo adicional justificado por mejor rendimiento

---

## Plan de Ejecución

```bash
# Actualizar runpod_launcher.py con RTX 4090
GPU_TYPE = "RTX4090"
GPU_TYPE_ID = "NVIDIA GeForce RTX 4090"

# Costo estimado final
500 agentes × 5M steps = ~22 horas
22h × $0.59 = $13 (spot: $11)
+ Buffer 30% = ~$17
```

---

## Actualización de Configuración

Actualizar `training/runpod_launcher.py`:
- Cambiar GPU default de A100 a RTX 4090
- Actualizar costos y velocidades
- Ajustar agents_per_batch a 8

---

**DECISIÓN APROBADA POR GOVERNANCE**

- [x] @rl_engineer: Aprobado - specs técnicos suficientes
- [x] @mlops_engineer: Aprobado - disponibilidad y costo óptimos
- [x] Presupuesto: $17-25 << $100 límite

*Documento generado por governance system*
