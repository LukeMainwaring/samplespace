#!/bin/bash

docker compose run --rm backend python -m alembic -c src/samplespace/alembic.ini upgrade head
