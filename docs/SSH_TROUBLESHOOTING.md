# Instrucciones para Claude Code: Diagnóstico SSH Exit Code 255

## Problema
Los comandos SSH fallan con exit code 255, que indica fallo de conexión, no del comando remoto.

## Pasos a seguir

### 1. Verificar conexión básica
```bash
ssh -p 1795 -o ConnectTimeout=10 -o BatchMode=yes root@193.183.22.54 "echo conexion_ok" 2>&1
```

### 2. Si funciona, usa comillas dobles para comandos remotos
```bash
# CORRECTO - comillas dobles:
ssh -p 1795 root@193.183.22.54 "pkill -f notify_complete || true"

# INCORRECTO - comillas simples con caracteres especiales:
ssh -p 1795 root@193.183.22.54 'pkill -f notify_complete || true'
```

### 3. Para crear archivos remotos, usa scp (NO heredoc a través de SSH)
```bash
# Paso 1: Crear archivo local
cat > /tmp/notify.sh << 'EOF'
#!/bin/bash
# contenido del script aquí
EOF

# Paso 2: Copiar al VPS
scp -P 1795 /tmp/notify.sh root@193.183.22.54:/workspace/

# Paso 3: Dar permisos
ssh -p 1795 root@193.183.22.54 "chmod +x /workspace/notify.sh"
```

### 4. Si la conexión básica falla, diagnóstico detallado
```bash
# Ver detalles del fallo
ssh -p 1795 -vvv root@193.183.22.54 "echo test" 2>&1 | head -50

# Verificar que el agente SSH tiene la clave
ssh-add -l
```

## Reglas generales

1. **Evita comillas simples externas** en comandos SSH complejos
2. **Usa `scp -P` (P mayúscula)** para transferir archivos en lugar de heredocs remotos
3. **Usa `ssh -p` (p minúscula)** para especificar puerto
4. **Siempre añade timeout** con `-o ConnectTimeout=10` para evitar bloqueos

## Conexión al VPS
- Host: 193.183.22.54
- Puerto: 1795
- Usuario: root
