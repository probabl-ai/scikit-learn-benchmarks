pixi run -e sklearn python -m sklbench --config configs/sklearn.json
pixi run -e sk-intel python -m sklbench --config configs/array-api-intel.json
pixi run -e intel python -m sklbench --config configs/sklearnex.json
