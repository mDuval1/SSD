MODEL_NAME = "vgg_ssd300_voc0712"
CONFIG_FILE="../SSD/configs/vgg_ssd300_voc0712.yaml"
# MODEL_NAME = "mobilenet_v2_ssd320_voc0712"
# CONFIG_FILE="../SSD/configs/mobilenet_v2_ssd320_voc0712.yaml"
LOG_STEPS=100
INIT_SIZE=1000
QUERY_SIZE=150
QUERY_STEP=30
TRAIN_STEPS_QUERIES=2500

nohup python train_active.py --config-file $CONFIG_FILE \
                    --log_step $LOG_STEPS \
                    --init_size $INIT_SIZE \
                    --query_size $QUERY_SIZE \
                    --query_step $QUERY_STEP \
                    --train_step_per_query $TRAIN_STEPS_QUERIES \
                    --strategy uncertainty_aldod_sampling > nohup_aldod.out &
                    
# nohup python train_active.py --config-file $CONFIG_FILE \
#                     --log_step $LOG_STEPS \
#                     --init_size $INIT_SIZE \
#                     --query_size $QUERY_SIZE \
#                     --query_step $QUERY_STEP \
#                     --train_step_per_query $TRAIN_STEPS_QUERIES \
#                     --strategy random_sampling > nohup_random.out &
