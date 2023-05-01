CUDA_VISIBLE_DEVICES=0
nvidia-smi

export project_dir=/home/ma-user/modelarts/user-job-dir/alpaca-lora
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
cd $project_dir

ls
cd peft
python setup.py develop
cd /data/chenjunsong/code/transformers
pip install -e .

cd /data/chenjunsong/download/
pip install bitsandbytes-0.38.0.post2-py3-none-any.whl
pip install accelerate
cd $project_dir

ln -s /data/chenjunsong/output/alpaca-lora ./output

run=$1
bs=64

#python finetune_bitfit.py ${run} --bs ${bs}
python finetune_difffit.py ${run} --bs ${bs}
