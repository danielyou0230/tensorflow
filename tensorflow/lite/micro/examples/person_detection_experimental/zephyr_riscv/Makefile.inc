ifeq ($(TARGET), zephyr_vexriscv)
	export ZEPHYR_TOOLCHAIN_VARIANT?=zephyr
	export TOOLCHAIN_BASE=${ZEPHYR_SDK_INSTALL_DIR}/riscv64-zephyr-elf/riscv64-zephyr-elf
	export TOOLCHAIN_VERSION=9.2.0
	PROJECT_INCLUDES += ${CURDIR} ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION} ${TOOLCHAIN_BASE}/include ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION}/riscv64-zephyr-elf/rv32i/ilp32
	ZEPHYR_PERSON_DETECTION_EXPERIMENTAL_SRCS = \
	tensorflow/lite/micro/examples/person_detection_experimental/detection_responder.cc \
	tensorflow/lite/micro/examples/person_detection_experimental/zephyr_riscv/src/image_provider.cc \
	tensorflow/lite/micro/examples/person_detection_experimental/main.cc \
	tensorflow/lite/micro/examples/person_detection_experimental/zephyr_riscv/src/main_functions.cc \
	tensorflow/lite/micro/examples/person_detection_experimental/model_settings.cc \
	tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_detect_model_data.cc \
	tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_image_data.cc \
	boards/litex_vexriscv.overlay \
	prj.conf

$(eval $(call generate_project,cmake,zephyr_cmake_project.cmake,person_detection_experimental,$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(ZEPHYR_PERSON_DETECTION_EXPERIMENTAL_SRCS) $(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(person_detection_HDRS),,$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS),$(CCFLAGS),))

$(PRJDIR)person_detection_experimental/cmake/CMakeLists.txt: $(PRJDIR)person_detection_experimental/cmake/zephyr_cmake_project.cmake
	@sed -E 's#\%\{INCLUDE_DIRS\}\%#$(PROJECT_INCLUDES)#g' $< > $@

#We are skipping here copy of `zephyr` third_party repository 
#To compile standalone project ZEPHYR_BASE enviroment variable should be set
person_detection_experimental_bin: generate_person_detection_experimental_cmake_project $(PRJDIR)person_detection_experimental/cmake/CMakeLists.txt
	( \
	  . ${ZEPHYR_BASE}/venv-zephyr/bin/activate; \
	  cmake -B${GENDIR}person_detection_experimental/build -DBOARD="litex_vexriscv" -H${PRJDIR}person_detection_experimental/cmake/ -DPython_ROOT_DIR=${ZEPHYR_BASE}/venv-zephyr/bin/; \
	  make -C ${GENDIR}person_detection_experimental/build; \
	)
endif
