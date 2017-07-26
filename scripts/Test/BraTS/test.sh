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

if [ "${PARAMS}" = "params_VNET"  ]
    then
#    srun  --gres=gpu:1 --pty  --mem=8G  python test_VNET.py -p"${PARAMS}"
#    srun  --gres=gpu:1,gmem:12GB --pty  --mem=8G  python test_VNET.py -p"${PARAMS}"
    srun  -p fast --pty  --mem=8G  python test_VNET.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_VNET_1"  ]
    then
    srun -p fast --gres=gpu:1  --pty  --mem=8G  python test_VNET_1.py -p"${PARAMS}"
#    srun    --pty  --mem=8G  python test_VNET_1.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_VNET_full"  ]
    then
    srun  --gres=gpu:1  --pty  --mem=8G  python test_VNET_full.py -p"${PARAMS}"
#    srun    --pty  --mem=8G  python test_VNET_1.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_two_pathways"  ]
    then
    KERAS_BACKEND=theano srun  --gres=gpu:1,gmem:12GB  --pty  --mem=20G  python test_two_pathways.py -p"${PARAMS}"
#    srun    --pty  --mem=8G  python test_VNET_1.py -p"${PARAMS}"

else
    echo 'ERROR in "test.sh"'
    echo 'Please, specify a valid parameter filename'
fi