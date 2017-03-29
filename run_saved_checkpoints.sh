SAVE_FOLDER='saved_networks/DDQN-SpaceInvaders-v0l_00025_evaluation/'

file_new=`ls -rt $SAVE_FOLDER | cut -d '.' -f 1 | cut -d '-' -f 3 | uniq ` 

for f in ${file_new[0]}
do
	python saved_model_run_v1.py --checkpoint $f
done