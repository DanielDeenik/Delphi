#!/bin/bash
# Import data for PG
python "$(dirname "$0")/import_pg.py" "$@"
