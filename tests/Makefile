# --- Delegator Makefile ---

test_all:
	$(MAKE) -C single test_all

test1 test2 test3 test4:
	$(MAKE) -C single $@

clean:
	$(MAKE) -C single clean
	$(MAKE) -C multi clean
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh
	$(MAKE) -C nas-benchmarks/IS clean
	$(MAKE) -C nas-benchmarks/CG clean
	$(MAKE) -C nas-benchmarks/MG clean

.PHONY: test_all test1 test2 test3 test4 clean
