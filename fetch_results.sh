
SUITE_RUN="train_3pc_1705446689"

EXP_NAME="mal"
SERVERS=("ec2-18-184-102-2.eu-central-1.compute.amazonaws.com" "ec2-18-184-240-77.eu-central-1.compute.amazonaws.com" "ec2-18-153-70-207.eu-central-1.compute.amazonaws.com")

RUN_ID=15

for HOST in {0..2}
do
  echo "Fetching $EXP_NAME run_$RUN_ID host_$HOST"
  SERVER=${SERVERS[$HOST]}
	LOCAL_PATH="/Users/hidde/PhD/auditing/cryptographic-auditing-mpc/doe-suite-results/$SUITE_RUN/$EXP_NAME/run_$RUN_ID/rep_0/server/host_$HOST"
	mkdir -p $LOCAL_PATH
	rsync -az "$SERVER:results/$SUITE_RUN/$EXP_NAME/run_$RUN_ID/rep_0/results/*" "$LOCAL_PATH"

	CONFIG_PATH="/Users/hidde/PhD/auditing/cryptographic-auditing-mpc/doe-suite-results/$SUITE_RUN/$EXP_NAME/run_$RUN_ID/rep_0/"
	rsync -az "$SERVER:results/$SUITE_RUN/$EXP_NAME/run_$RUN_ID/rep_0/config.json" "$CONFIG_PATH"
done
