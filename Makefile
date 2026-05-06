# SST Voice Pipecat - Docker Build & Deploy
# ==========================================

# Configuration
IMAGE_NAME ?= sst-voice-pipecat
IMAGE_TAG ?= latest
REGISTRY ?=
CONTAINER_NAME ?= sst-voice-pipecat
PORT ?= 7860

# Full image name
ifdef REGISTRY
	FULL_IMAGE = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
else
	FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
endif

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m # No Color

.PHONY: help build run stop logs shell push deploy clean rebuild

## Help
help:
	@echo "$(GREEN)SST Voice Pipecat - Available commands:$(NC)"
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@echo "  $(YELLOW)build$(NC)      - Build Docker image"
	@echo "  $(YELLOW)rebuild$(NC)    - Rebuild Docker image (no cache)"
	@echo "  $(YELLOW)run$(NC)        - Run container locally"
	@echo "  $(YELLOW)stop$(NC)       - Stop running container"
	@echo "  $(YELLOW)logs$(NC)       - View container logs"
	@echo "  $(YELLOW)shell$(NC)      - Open shell in running container"
	@echo "  $(YELLOW)push$(NC)       - Push image to registry"
	@echo "  $(YELLOW)deploy$(NC)     - Build and push to registry"
	@echo "  $(YELLOW)clean$(NC)      - Remove container and image"
	@echo ""
	@echo "$(GREEN)Docker Compose Commands:$(NC)"
	@echo "  $(YELLOW)up$(NC)         - Start services (background)"
	@echo "  $(YELLOW)up-fg$(NC)      - Start services (foreground)"
	@echo "  $(YELLOW)up-build$(NC)   - Rebuild and start services"
	@echo "  $(YELLOW)down$(NC)       - Stop services"
	@echo "  $(YELLOW)dc-logs$(NC)    - View compose logs"
	@echo "  $(YELLOW)dc-status$(NC)  - Show compose status"
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  IMAGE_NAME=$(IMAGE_NAME)"
	@echo "  IMAGE_TAG=$(IMAGE_TAG)"
	@echo "  REGISTRY=$(REGISTRY)"
	@echo "  PORT=$(PORT)"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make up                  # Start with docker compose"
	@echo "  make up-build            # Rebuild and start"
	@echo "  make build && make run   # Build and run manually"
	@echo "  make push REGISTRY=docker.io/myuser IMAGE_TAG=v1.0"

## Build Docker image
build:
	@echo "$(GREEN)Building Docker image: $(FULL_IMAGE)$(NC)"
	docker build -t $(FULL_IMAGE) .

## Rebuild Docker image without cache
rebuild:
	@echo "$(GREEN)Rebuilding Docker image (no cache): $(FULL_IMAGE)$(NC)"
	docker build --no-cache -t $(FULL_IMAGE) .


## View container logs
logs:
	docker logs -f $(CONTAINER_NAME)

## View last 100 lines of logs
logs-tail:
	docker logs --tail 100 -f $(CONTAINER_NAME)

## Open shell in running container
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

## Push image to registry
push:
ifndef REGISTRY
	$(error REGISTRY is not set. Usage: make push REGISTRY=your-registry.com)
endif
	@echo "$(GREEN)Pushing image to registry: $(FULL_IMAGE)$(NC)"
	docker push $(FULL_IMAGE)

## Build and push to registry
deploy: build push
	@echo "$(GREEN)Deployment complete: $(FULL_IMAGE)$(NC)"

## Tag and push with version
release:
ifndef VERSION
	$(error VERSION is not set. Usage: make release VERSION=1.0.0)
endif
	@echo "$(GREEN)Creating release: $(VERSION)$(NC)"
	docker tag $(IMAGE_NAME):latest $(IMAGE_NAME):$(VERSION)
ifdef REGISTRY
	docker tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
endif

# ===========================================
# Docker Compose Commands
# ===========================================

## Start with docker compose
restart:
	@echo "$(GREEN)Starting services with docker compose$(NC)"
	docker compose restart
## Start with docker compose
up:
	@echo "$(GREEN)Starting services with docker compose$(NC)"
	docker compose up -d

## Start with docker compose (foreground)
up-fg:
	@echo "$(GREEN)Starting services with docker compose (foreground)$(NC)"
	docker compose up

## Stop docker compose services
down:
	@echo "$(YELLOW)Stopping docker compose services$(NC)"
	docker compose down

## Rebuild and start with docker compose
up-build:
	@echo "$(GREEN)Rebuilding and starting services with docker compose$(NC)"
	docker compose up -d --build

## View docker compose logs
dc-logs:
	docker compose logs -f

## docker compose status
dc-status:
	docker compose ps
