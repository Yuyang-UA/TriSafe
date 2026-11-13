TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=/path/to/ego_status_mlp.ckpt
CACHE_PATH=YOUR_PATH_TO_METRIC_CACHE

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=ego_mlp_agent \
metric_cache_path=$CACHE_PATH \
