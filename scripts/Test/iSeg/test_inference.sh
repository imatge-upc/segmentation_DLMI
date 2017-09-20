#!/usr/bin/env bash

for i in "$@"
do
case $i in
    -p=*|--params=*)
    PARAMS="${i#*=}"
    shift # past argument=value
    ;;
    *)
            # unknown option
    ;;
esac
done

if [ "${PARAMS}" = "params_full"  ]
    then

    srun  -p fast --gres=gpu:1 --pty  --mem=8G  python test_inference_VNET.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_patches"  ]
    then
    srun  -p fast --gres=gpu:1  --pty  --mem=8G  python test_inference_VNET.py -p"${PARAMS}"
else
    echo 'ERROR in "test_inference.sh"'
    echo 'Please, specify a valid parameter filename'
fi