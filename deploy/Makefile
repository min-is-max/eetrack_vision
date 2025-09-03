REG=ghcr.io/<ORG>         # ← 바꿔 넣기 (ex. liberty-robotics)
IMG=$(REG)/wld
TAG=runtime-2.1.0-cu124

build:
	DOCKER_BUILDKIT=1 docker build -f deploy/Dockerfile.runtime -t $(IMG):$(TAG) .

push:
	docker push $(IMG):$(TAG)

run-bash:
	docker run --gpus all --rm -it --shm-size=16g \
	  -v $$PWD/sample_data:/data \
	  -v $$PWD/outputs:/outputs \
	  $(IMG):$(TAG) bash

smoke:
	bash deploy/smoke_test.sh $(IMG):$(TAG)
