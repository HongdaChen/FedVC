
# an example
python experiments.py --model=ConvNet --dataset=cifar10 --alg=fedprox --lr=0.01 --batch_size=64 --epochs=10 --n_parties=10 --mu=0.1 --comm_round=20 --partition=noniid-labeldir --beta=0.5 --datadir=../data/data152754 --logdir=paddle_ --noise=0 --init_seed=0

# fedavg cifar10 
seed=0
for beta in 0.5 0.1
do
  for alg in fedavg
  do
    python experiments.py --model=ConvNet \
      --dataset=cifar10 \
      --alg=$alg \
      --lr=0.01 \
      --batch_size=64 \
      --epochs=10 \
      --n_parties=10 \
      --comm_round=20 \
      --partition=noniid-labeldir \
      --beta=$beta \
      --datadir='../data/data152754' \
      --logdir='../logs' \
      --init_seed=$seed
  done
done

# fedprox cifar10 
seed=0
for beta in 0.5 0.1
do
  for alg in fedprox
  do
    for mu in 0.001 0.01 0.1 1
    do
        python experiments.py --model=ConvNet \
        --dataset=cifar10 \
        --alg=$alg \
        --lr=0.01 \
        --batch_size=64 \
        --epochs=10 \
        --n_parties=10 \
        --comm_round=20 \
        --partition=noniid-labeldir \
        --beta=$beta \
        --mu=$mu \
        --datadir='../data/data152754' \
        --logdir='../logs' \
        --init_seed=$seed
    done
  done
done

# moon cifar10 
seed=0
for beta in 0.5 0.1
do
  for alg in moon
  do
    for mu in 0.1 1 5 10
    do
        python experiments.py --model=ConvNet \
        --dataset=cifar10 \
        --alg=$alg \
        --lr=0.01 \
        --batch_size=64 \
        --epochs=10 \
        --n_parties=10 \
        --comm_round=20 \
        --partition=noniid-labeldir \
        --beta=$beta \
        --mu=$mu \
        --datadir='../data/data152754' \
        --logdir='../logs' \
        --init_seed=$seed
    done
  done
done

