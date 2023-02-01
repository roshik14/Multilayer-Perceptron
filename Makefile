.PHONY: all tests clean install dist dvi uninstall run

CXX = g++
CXXFLAGS = -lgtest -std=c++17
CHECKFLAGS = -Wall -Werror -Wextra
OS:=$(shell uname -s)
TEXI2HTML = makeinfo --no-split --html --no-warn

SOURCES = Controller/*.cc Model/*.cc View/*.cc *.cc
HEADERS = Controller/*.h Model/*.h *.h View/*.h

TEST_SOURCES = Model/datasetreader.cc Model/neuralnetwork.cc Model/graphnetwork.cc Model/matrixnetwork.cc Model/graphneuron.cc Model/weights.cc Model/weightsreader.cc Model/weightssaver.cc tests.cc

BUILD_SRC = Controller Model View main.cc *.h MLP.pro

SUP = --suppress=

all: install run

install:
	mkdir ../build
	cp -r $(BUILD_SRC) ../build/
	cd ../build; qmake -makefile MLP.pro
	make -C ../build
	make clean -C ../build

uninstall:
	@rm -rf ../build
	@echo "Uninstalled"

clean:
	@/bin/rm -rf *.o *.out moc_* info.html *.tar tests

dist: 
	@tar -zcf MLP.tar * Makefile *.tex

dvi: info.tex
	$(TEXI2HTML) info.tex
	open info.html

tests: tests.cc
	@$(CXX) $(CHECKFLAGS) $(TEST_SOURCES) -o tests $(CXXFLAGS)
	@./tests

cppcheck:
	@cppcheck --enable=all --std=c++17 --language=c++ $(SUP)unusedFunction $(SUP)missingInclude $(SUP)unmatchedSuppression $(SOURCES)

style:
	@cp -r ../materials/linters/.clang-format ./
	@clang-format -n $(SOURCES) $(HEADERS)
	@rm -rf .clang-format

run:
ifeq ($(OS), Darwin)
	./../build/MLP.app/Contents/MacOS/MLP
else
	./../build/MLP
endif

