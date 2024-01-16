

script?=audit_owner_unlearn
dataset?=mnist_6k_4party
protocol?=rep4-ring

RING_64=-R 64

AUDITARGS:= dataset__$(dataset) audit_trigger_idx__1



LINK_UTILS=$(PWD)/MP-SPDZ/Compiler/script_utils

LINK=$(PWD)/MP-SPDZ/Programs/Source/$(script).mpc

test?=test_utils
TEST=$(PWD)/MP-SPDZ/Programs/Source/$(test).mpc


all: emulate


clean:
	cd MP-SPDZ && $(MAKE) clean

setup-mpspdz:
	cd MP-SPDZ && $(MAKE) -j 8 tldr
	cd MP-SPDZ && make emulate.x

# only if simlink does not exist, create it
simlink:
	[ -L $(LINK_UTILS) ] && [ -e $(LINK_UTILS) ] || ln -s  $(PWD)/script_utils $(LINK_UTILS)
	[ -L $(LINK) ] && [ -e $(LINK) ] || ln -s  $(PWD)/scripts/$(script).mpc $(LINK)



compile-debug: simlink
	cd MP-SPDZ && ./compile.py $(RING_64) $(script) $(AUDITARGS) emulate__True debug__True

compile: simlink
	cd MP-SPDZ && ./compile.py $(RING_64) $(script) --budget 10000 $(AUDITARGS) emulate__True debug__False

emulate: compile
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-emulate__True-debug__False

emulate-debug: compile-debug
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-emulate__True-debug__True

ff4: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E rep4-ring $(RING_64) -Z 4 $(script) $(AUDITARGS) debug__False

ff4-debug: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E rep4-ring $(RING_64) -Z 4 $(script) $(AUDITARGS) debug__True

protocol: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(AUDITARGS) debug__False

protocol-debug: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) -C --budget 10000 $(AUDITARGS) debug__True

train-debug: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E rep4-ring $(RING_64) -Z 4 $(script) 10 16 -- -v

ring: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(RING_64) -CD -Z 3 --budget 1000 -C $(AUDITARGS) emulate__True debug__False

ring-mal: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(RING_64) --budget 1000 -C $(AUDITARGS) emulate__True debug__False


ring-test: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(test) $(RING_64) -Y --budget 1000 -C $(AUDITARGS) emulate__True debug__False

ring-sha: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(RING_64) -CD -Z 3 --budget 1000 -C $(AUDITARGS) emulate__True debug__False consistency_check__sha3


compile-field: simlink
	cd MP-SPDZ && ./compile.py $(script) $(AUDITARGS) emulate__True debug__True

compile-field-256: simlink
	cd MP-SPDZ && ./compile.py -F 256 $(script) --budget 10000 $(AUDITARGS) emulate__True debug__False

field-256: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -F 256 -E $(protocol) $(script) $(AUDITARGS) debug__False -- -lgp 256

field: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -Y -C --budget 10000 -E $(protocol) $(script) $(AUDITARGS) emulate__False debug__False batch_size__128 -- -lgp 128 -v -b 50

field-std: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -C --budget 10000 -E $(protocol) $(script) $(AUDITARGS) emulate__False debug__False -- -lgp 128 -v

field-bls377: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -F 251 -E $(protocol) $(script) $(AUDITARGS) debug__False -- -P 8444461749428370424248824938781546531375899335154063827935233455917409239041

field-bls377-ped: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -F 251 -E $(protocol) $(script) $(AUDITARGS) consistency_check__cerebro debug__False -- -P 8444461749428370424248824938781546531375899335154063827935233455917409239041

deb: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(RING_64) -Z 3 --budget 1000 -C audit_trigger_idx__0 batch_size__128 consistency_check__False dataset__mnist_full_3party debug__False emulate__False n_input_parties__3 n_threads__36 trunc_pr__True



plots:
	cd doe-suite && $(MAKE) etl-super config=consistency out=$(out)
	cd doe-suite && $(MAKE) etl-super config=inference out=$(out)
	cd doe-suite && $(MAKE) etl-super config=train out=$(out)