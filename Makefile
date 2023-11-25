

script?=audit_owner_unlearn
dataset?=mnist_6k_4party
protocol?=rep4-ring

RING_64=-R 64

AUDITARGS:= dataset__$(dataset) audit_trigger_idx__1


LINK_UTILS=$(PWD)/MP-SPDZ/Compiler/script_utils

LINK=$(PWD)/MP-SPDZ/Programs/Source/$(script).mpc

all: emulate


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
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(RING_64) $(AUDITARGS) emulate__True debug__False

compile-field: simlink
	cd MP-SPDZ && ./compile.py $(script) $(AUDITARGS) emulate__True debug__True

compile-field-256: simlink
	cd MP-SPDZ && ./compile.py -F 256 $(script) --budget 10000 $(AUDITARGS) emulate__True debug__False

field-256: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -F 256 -E $(protocol) $(script) $(AUDITARGS) debug__False -- -lgp 256

field: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -C --budget 10000 -E $(protocol) $(script) $(AUDITARGS) emulate__False debug__True -- -lgp 128

field-bls377: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -F 251 -E $(protocol) $(script) $(AUDITARGS) debug__False -- -P 8444461749428370424248824938781546531375899335154063827935233455917409239041

field-bls377-slow: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -P 8444461749428370424248824938781546531375899335154063827935233455917409239041 -F 64 -E $(protocol) $(script) $(AUDITARGS) debug__False

field: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -C --budget 10000 -E $(protocol) $(script) $(AUDITARGS) emulate__False debug__True batch_size__16 -- -lgp 128
