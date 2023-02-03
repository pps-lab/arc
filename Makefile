

script?=audit_owner_unlearn
dataset?=mnist_6k_4party

MPSPDZFLAGS=-R 64

AUDITARGS:= dataset__$(dataset) emulate__True


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
	cd MP-SPDZ && ./compile.py $(MPSPDZFLAGS) $(script) $(AUDITARGS) debug__True

compile: simlink
	cd MP-SPDZ && ./compile.py $(MPSPDZFLAGS) $(script) $(AUDITARGS) debug__False

emulate: compile
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-debug__False

emulate-debug: compile-debug
	cd MP-SPDZ && ./emulate.x $(script)-$(subst $e ,-,$(AUDITARGS))-debug__True