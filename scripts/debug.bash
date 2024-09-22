
python -m debugpy --listen 50678  --wait-for-client eval.py --model-folder runs/rvt2  --eval-datafolder ./data/test --log-name test/1 --device 0 --model-name model_last.pth

python -m debugpy --listen 50678  --wait-for-client train.py --exp_cfg_path configs/rvt2.yaml --mvt_cfg_path model/mvt/configs/rvt2.yaml