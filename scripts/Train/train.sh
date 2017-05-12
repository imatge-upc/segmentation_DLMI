#!/usr/bin/env bash
for i in "$@"
do
case $i in
    -m=*|--model=*)
    MODEL="${i#*=}"
    shift # past argument=value
    ;;
    -f=*|--filename=*)
    FILENAME="${i#*=}"
    shift # past argument=value
    ;;
    -w=*|--weights=*)
    WEIGHTS="${i#*=}"
    shift # past argument=value
    ;;
    *)
            # unknown option
    ;;
esac
done

: ${WEIGHTS:=-1}

if [ "${FILENAME}" = "two_pathways"  ]
    then
    THEANO_FLAGS=device=gpu,optimizer_including=cudnn srun --gres=gpu:1,gmem:4GB --mem=20G --pty python train.py -f"${FILENAME}" -m"${MODEL}" -w"${WEIGHTS}"
else
    THEANO_FLAGS=device=gpu,optimizer_including=cudnn srun --gres=gpu:1,gmem:6GB --mem=20G --pty python train.py -f"${FILENAME}" -m"${MODEL}" -w"${WEIGHTS}"
fi