#!/bin/bash
# Deployment script for NautilusTrader RL Agents
# Run this on the VPS after copying files

set -e

echo "=========================================="
echo "NautilusTrader RL Agents - Deployment"
echo "=========================================="

# Create directory structure
echo "Creating directory structure..."
sudo mkdir -p /opt/nautilus-agents
sudo chown -R $USER:$USER /opt/nautilus-agents

# Copy files (assumes files are in current directory)
echo "Copying files..."
cp -r . /opt/nautilus-agents/

# Create additional directories
mkdir -p /opt/nautilus-agents/models
mkdir -p /opt/nautilus-agents/logs
mkdir -p /opt/nautilus-agents/data/catalog

# Set up environment file
if [ ! -f /opt/nautilus-agents/.env ]; then
    echo "Creating .env file from template..."
    cp /opt/nautilus-agents/.env.example /opt/nautilus-agents/.env
    echo "IMPORTANT: Edit /opt/nautilus-agents/.env with your credentials!"
fi

# Build Docker images
echo "Building Docker images..."
cd /opt/nautilus-agents
docker-compose build

echo "=========================================="
echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Edit /opt/nautilus-agents/.env with your credentials"
echo "2. Start with: docker-compose up -d"
echo "3. Check logs: docker-compose logs -f"
echo "=========================================="
