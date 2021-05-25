spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 3 \
    --py-files assignment2.py \
    --output $1 
