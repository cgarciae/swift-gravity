N_OBJECTS_VALUES="10 100 500 1000 1500 2000 2500"
DEVICE="gpu"
STEPS=500


swift run Benchmarks tf \
    --n-objects-values "$N_OBJECTS_VALUES" \
    --device "$DEVICE" \
    --steps "$STEPS" \
    "$@"