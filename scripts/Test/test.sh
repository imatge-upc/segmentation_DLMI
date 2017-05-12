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
    -n=*|--nsubjects=*)
    NSUBJECTS="${i#*=}"
    shift # past argument=value
    ;;
    -w=*|--weights=*)
    WEIGHTS="${i#*=}"
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
            # unknown option
    ;;
esac
done

if [ "$FILENAME" = "two_pathways"  ]
    then
    THEANO_FLAGS=device=gpu,optimizer_including=cudnn srun  --gres=gpu:1 --mem=4G  --pty python test.py -filename="${FILENAME}" -model="${MODEL}" -nsubjects=${NSUBJECTS}   -weights="${WEIGHTS}" -listit=1
elif [ "$FILENAME" = "u_net"  ]
then
    THEANO_FLAGS=device=gpu,optimizer_including=cudnn srun   --gres=gpu:1  --mem=4G --pty python test.py -filename="${FILENAME}" -model="${MODEL}" -nsubjects=${NSUBJECTS}  -weights="${WEIGHTS}" -listit=1
else
    THEANO_FLAGS=device=gpu,optimizer_including=cudnn srun   --gres=gpu:1  --mem=4G --pty python test.py -filename="${FILENAME}" -model="${MODEL}" -nsubjects=${NSUBJECTS} -weights="${WEIGHTS}" -listit=1
fi


