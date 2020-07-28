CONFIG_FILE_BIGGER="../SSD/configs/vgg_ssd300_voc0712.yaml"
CONFIG_FILE="configs/mobilenet_v2_ssd320_voc0712.yaml"
LOG_STEPS=100
INIT_SIZE=1000
QUERY_SIZE=150
QUERY_STEP=30
TRAIN_STEPS_QUERIES=2500
PREVIOUS_QUERIES="outputs/mobilenet_v2_ssd320_voc0712/results/uncertainty_aldod_sampling/experiment-20200701095400/queries.txt"


nohup python train_active.py --config-file $CONFIG_FILE \
                    --log_step $LOG_STEPS \
                    --init_size $INIT_SIZE \
                    --query_size $QUERY_SIZE \
                    --query_step $QUERY_STEP \
                    --train_step_per_query $TRAIN_STEPS_QUERIES \
                    --previous_queries $PREVIOUS_QUERIES \
                    --strategy from_save > nohup_fromsave.out &


nohup python train_active.py --config-file $CONFIG_FILE_BIGGER \
                    --log_step $LOG_STEPS \
                    --init_size $INIT_SIZE \
                    --query_size $QUERY_SIZE \
                    --query_step $QUERY_STEP \
                    --train_step_per_query $TRAIN_STEPS_QUERIES \
                    --previous_queries $PREVIOUS_QUERIES \
                    --strategy from_save > nohup_fromsave_bigger.out &