
SUITE_RUN="audit_consistency_1705060337"

EXP_NAME="audit_sample_knnshapley_mal"
SERVERS=("ec2-18-192-119-70.eu-central-1.compute.amazonaws.com" "ec2-3-64-178-175.eu-central-1.compute.amazonaws.com" "ec2-18-193-112-141.eu-central-1.compute.amazonaws.com")

RUN_ID=17

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
