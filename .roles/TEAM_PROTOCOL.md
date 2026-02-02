# Team Protocol - Autonomous Operation

> **MODO AUTÓNOMO ACTIVADO**
> El sistema toma decisiones automáticamente según `config/autonomous_config.yaml`
> Los roles ejecutan sin pedir aprobación. Solo alertan si es CRÍTICO.
> El usuario supervisa resultados, no aprueba decisiones.

## Operating Principles

1. **ACT FIRST**: Ejecutar según config, no preguntar
2. **LOG EVERYTHING**: Todas las decisiones se documentan
3. **ALERT ONLY CRITICAL**: Solo Telegram si es crítico
4. **SELF-HEAL**: Si algo falla, intentar recuperar automáticamente
5. **LEARN**: Cada error va a LESSONS_LEARNED.md

## How This System Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COLLABORATION FLOW                              │
│                                                                      │
│   ┌──────────┐     DECISION_QUEUE.md      ┌──────────┐             │
│   │ RL Eng   │◄──────────────────────────►│ Quant Dev│             │
│   └────┬─────┘                            └────┬─────┘             │
│        │         ┌──────────────┐              │                   │
│        └────────►│ DECISIONS.md │◄─────────────┘                   │
│                  │ (decisions   │                                  │
│                  │  log)        │                                  │
│                  └──────┬───────┘                                  │
│                         │                                          │
│                         ▼                                          │
│                  ┌──────────────┐                                  │
│                  │ MLOps Eng    │                                  │
│                  └──────────────┘                                  │
│                         │                                          │
│                         ▼                                          │
│              RETROSPECTIVES.md                                     │
│              (continuous improvement)                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Starting a Session

When you start working, Claude should:

1. **Read this file** to understand the protocol
2. **Check `DECISION_QUEUE.md`** for pending decisions
3. **Read the relevant role file** (e.g., `rl_engineer/STATE.md`)
4. **Check if another role needs input** before proceeding

## Role Handoff Protocol

When a role completes work that affects another role:

```markdown
## HANDOFF: [FROM_ROLE] → [TO_ROLE]
- **Date**: YYYY-MM-DD
- **Context**: Brief description
- **Files Changed**: list of files
- **Decision Needed**: yes/no
- **Blocking**: yes/no
```

## Decision Types

| Type | Who Decides | Process |
|------|-------------|---------|
| **Technical** | Single role | Document in role's STATE.md |
| **Cross-Role** | 2+ roles | Add to DECISION_QUEUE.md |
| **Architectural** | All roles | Requires team review |
| **Urgent** | Any role | Can decide, but must document rationale |

## Invoking Roles

User can say:
- `"@rl_engineer continúa con validation"` → Loads RL Engineer context
- `"@quant_developer revisa la estrategia"` → Loads Quant Dev context
- `"@team_review necesito decisión sobre X"` → Multi-role discussion
- `"@retrospective qué aprendimos esta semana"` → Improvement review

## Continuous Improvement Loop

```
   PLAN → DO → CHECK → ACT
     │     │      │      │
     │     │      │      └── Update LESSONS_LEARNED.md
     │     │      └── Review in RETROSPECTIVES.md
     │     └── Work documented in role STATE.md
     └── Decisions in DECISION_QUEUE.md
```

## File Structure

```
.roles/
├── TEAM_PROTOCOL.md      # This file - how to collaborate
├── DECISION_QUEUE.md     # Pending decisions needing input
├── DECISIONS.md          # Log of all decisions made
├── LESSONS_LEARNED.md    # What we've learned (improvements)
│
├── rl_engineer/
│   ├── STATE.md          # Current state, blockers, next actions
│   └── EXPERTISE.md      # What this role knows/can do
│
├── quant_developer/
│   ├── STATE.md
│   └── EXPERTISE.md
│
├── mlops_engineer/
│   ├── STATE.md
│   └── EXPERTISE.md
│
└── retrospectives/
    └── YYYY-MM-DD.md     # Weekly/sprint retrospectives
```
