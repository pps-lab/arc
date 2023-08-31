

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
	cd MP-SPDZ && ./compile.py $(RING_64) $(script) $(AUDITARGS) emulate__True debug__False

emulate: compile
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-emulate__True-debug__False

emulate-debug: compile-debug
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-emulate__True-debug__True

ff4: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E rep4-ring $(RING_64) -Z 4 $(script) $(AUDITARGS) debug__False

ff4-debug: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E rep4-ring $(RING_64) -Z 4 $(script) $(AUDITARGS) debug__True

protocol:
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(AUDITARGS) debug__False

protocol-debug:
	cd MP-SPDZ && ./Scripts/compile-run.py -E $(protocol) $(script) $(AUDITARGS) debug__True

train-debug: simlink
	cd MP-SPDZ && ./Scripts/compile-run.py -E rep4-ring $(RING_64) -Z 4 $(script) 10 16 -- -v

