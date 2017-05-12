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
    -s=*|--slist=*)
    SUBJLIST="${i#*=}"
    shift # past argument=value
    ;;
    -S*|--store*)
    STORE=$1
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

THEANO_FLAGS=device=gpu,optimizer_including=cudnn srun  --gres=gpu:1 --mem=4G --pty python test_training.py -filename="${FILENAME}" -model="${MODEL}" -nsubjects=${NSUBJECTS} -slist="${SUBJLIST}" -store=1 -weights="${WEIGHTS}"


