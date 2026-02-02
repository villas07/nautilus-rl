# Decision Queue

> **MODO AUTÓNOMO ACTIVADO**
> Las decisiones se toman automáticamente según `config/autonomous_config.yaml`
> Solo items CRÍTICOS requieren intervención humana.

---

## RESOLVED (Auto-configured)

### ~~DQ-001: Validation Thresholds~~ → RESOLVED
- **Resolution**: Configured in `autonomous_config.yaml`
- **Values**: Sharpe > 1.5, MaxDD < 15%, WinRate > 50%
- **Auto-action**: Failed agents archived automatically

### ~~DQ-002: Position Sizing~~ → RESOLVED
- **Resolution**: Confidence-scaled positioning
- **Values**: Base $1000, scales 50%-150% by confidence
- **Auto-action**: Skip trades below 60% confidence

### ~~DQ-003: Training Infrastructure~~ → RESOLVED
- **Resolution**: Hybrid mode (local dev, RunPod production)
- **Auto-trigger**: >10 agents or >10M timesteps → RunPod

---

## PENDING (None)

_No hay decisiones pendientes. El sistema opera autónomamente._

---

## ESCALATION CRITERIA

Solo escalar a humano si:
1. **Budget exceeded**: Costos superan límites definidos
2. **System failure**: Error no recuperable
3. **Security issue**: Credenciales comprometidas
4. **Novel situation**: Algo no cubierto en config

Para escalar, crear archivo: `.roles/ESCALATION_YYYY-MM-DD.md`
