# RapidOCR Docker Dev Environments
COMPOSE_FILE := docker/docker-compose.yaml
ENGINES := onnxruntime-cpu onnxruntime-gpu tensorrt paddle openvino pytorch mnn

.PHONY: help build-all clean $(foreach e,$(ENGINES),build-$(e) test-$(e) shell-$(e))

help:
	@echo "RapidOCR Docker Dev Environments"
	@echo ""
	@echo "Usage:"
	@echo "  make build-{engine}    Build a dev image"
	@echo "  make test-{engine}     Run tests in container"
	@echo "  make shell-{engine}    Open bash shell in container"
	@echo "  make build-all         Build all images"
	@echo "  make clean             Remove all containers and images"
	@echo ""
	@echo "Engines: $(ENGINES)"
	@echo ""
	@echo "Examples:"
	@echo "  make build-tensorrt"
	@echo "  make test-onnxruntime-cpu"
	@echo "  make shell-pytorch"

build-%:
	docker compose -f $(COMPOSE_FILE) build $*

test-%:
	docker compose -f $(COMPOSE_FILE) run --rm $* pytest tests/ -v

shell-%:
	docker compose -f $(COMPOSE_FILE) run --rm $* bash

build-all:
	docker compose -f $(COMPOSE_FILE) build

clean:
	docker compose -f $(COMPOSE_FILE) down --rmi local --volumes
