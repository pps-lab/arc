

script?=audit_owner_unlearn

MPSPDZFLAGS=-R 64

AUDITARGS=dataset__mnist


LINK=$(PWD)/MP-SPDZ/Programs/Source/$(script).mpc


all: emulate

# only if simlink does not exist, create it
simlink:
	[ -L $(LINK) ] && [ -e $(LINK) ] || ln -s  $(PWD)/scripts/$(script).mpc $(LINK)


compile-debug: simlink
	cd MP-SPDZ && ./compile.py $(MPSPDZFLAGS) $(script) $(AUDITARGS) debug__True

compile: simlink
	cd MP-SPDZ && ./compile.py $(MPSPDZFLAGS) $(script) $(AUDITARGS) debug__False

emulate: compile
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-debug__False

emulate-debug: compile
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-debug__True