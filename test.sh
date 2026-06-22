pixi run -e sklearn python -m sklbench --config configs/sklearn.json
pixi run -e skl-cpu python -m sklbench --config configs/array-api-cpu.json
# pixi run -e skl-intel python -m sklbench --config configs/array-api-intel.json
pixi run -e intel python -m sklbench --config configs/sklearnex.json
