ROOT=`pwd`
dataset=pinterest
model=VBPR
cnn=resnet
lr='[0.01,1e-4,1e-3]'
regs='[1e-1,1e-3,0]'
eps='0.1'
lmd='1.'
prefix='best'

python -u $ROOT/src/main.py --adv=True --model=${model} --dataset=$ROOT/data/${dataset} --weight_dir=./weights --lr=${lr} --regs=${regs} --epsilon=${eps} --lmd=${lmd} --emb1_K=64 --verbose=50  --batch_size=512 --cnn=${cnn} --epoch=2000 > "$ROOT/logs/${prefix}-${model}-${dataset}-${cnn}-lr*${lr}*-regs*${regs}*-eps*${eps}*-lmd*${lmd}*.log"
