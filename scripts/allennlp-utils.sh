function train(){
    SER_DIR=`readlink -m ${2:-/tmp/models}/$1`
    echo "Saving to $SER_DIR."
    allennlp train "experiments/$1.jsonnet" -s $SER_DIR
}

function evaluate(){
    allennlp evaluate $1/model.tar.gz $2 --cuda-device=0
}

function predict(){
    PREDICTOR=${3:-text_classifier}
    OUTPUT_FILE=`echo $2 | sed "s/.*\//predictions_/g" `
    echo "Saving predictions to /tmp/$1/$OUTPUT_FILE."
    allennlp predict $1/model.tar.gz $2 --output-file /tmp/$1/$OUTPUT_FILE \
            --predictor $PREDICTOR \
            --cuda-device=0 \
            --include-package rationales \
            --silent
}