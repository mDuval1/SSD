CONFIG_FILE="../SSD/configs/mobilenet_v2_ssd320_voc0712.yaml"
LOG_STEPS=10
INIT_SIZE=1000
QUERY_SIZE=100
QUERY_STEP=30
TRAIN_STEPS_QUERIES=1000

# rm -rf outputs

nohup python train_active.py --config-file $CONFIG_FILE \
                    --log_step $LOG_STEPS \
                    --init_size $INIT_SIZE \
                    --query_size $QUERY_SIZE \
                    --query_step $QUERY_STEP \
                    --train_step_per_query $TRAIN_STEPS_QUERIES \
                    --strategy uncertainty_aldod_sampling > nohup_aldod.out &
                    
nohup python train_active.py --config-file $CONFIG_FILE \
                    --log_step $LOG_STEPS \
                    --init_size $INIT_SIZE \
                    --query_size $QUERY_SIZE \
                    --query_step $QUERY_STEP \
                    --train_step_per_query $TRAIN_STEPS_QUERIES \
                    --strategy random_sampling > nohup_random.out &
