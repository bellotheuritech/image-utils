PROJECT_NAME := image-utils
LATEST := $(PROJECT_NAME):latest

all: docker-dev lint

docker:
	docker build -t $(LATEST) .

docker-dev:
	docker build -f Dockerfile.dev -t $(LATEST) .

lint:
	./run_pylint.sh $(LATEST)

functest :
	docker run -it $(LATEST)

pytest:
	./run_pytest.sh $(LATEST)

.PHONY: docker lint pytest functest test
