IMAGE_NAME=dcpcr
TAG=latest
CODE= $(shell pwd)

USER_ID= $(shell id -u)
GROUP_ID= $(shell id -g)
# CONFIG:=$(shell realpath ${CONFIG})
build:
	@echo Building docker container $(IMAGE_NAME)
	docker build -t $(IMAGE_NAME):$(TAG) --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) .

test:
	@echo NVIDIA and CUDA setup
	@nvidia-docker run --rm $(IMAGE_NAME):$(TAG) nvidia-smi
	@echo PytTorch CUDA setup installed?
	@nvidia-docker run --rm $(IMAGE_NAME):$(TAG) python3 -c "import torch; print(torch.cuda.is_available())"
