
SUITE_RUN="audit_knnshapley_bert_mal_1717669576"

EXP_NAME="audit_sample_knnshapley"
SERVERS=("ec2-35-158-3-104.eu-central-1.compute.amazonaws.com" "ec2-35-158-3-104.eu-central-1.compute.amazonaws.com" "ec2-35-158-3-104.eu-central-1.compute.amazonaws.com")

RUN_ID=3

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
