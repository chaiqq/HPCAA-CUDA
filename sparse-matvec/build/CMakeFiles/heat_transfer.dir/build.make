# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/build

# Include any dependencies generated for this target.
include CMakeFiles/heat_transfer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/heat_transfer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/heat_transfer.dir/flags.make

CMakeFiles/heat_transfer.dir/binding/interface.cpp.o: CMakeFiles/heat_transfer.dir/flags.make
CMakeFiles/heat_transfer.dir/binding/interface.cpp.o: ../binding/interface.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/heat_transfer.dir/binding/interface.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/heat_transfer.dir/binding/interface.cpp.o -c /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/binding/interface.cpp

CMakeFiles/heat_transfer.dir/binding/interface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/heat_transfer.dir/binding/interface.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/binding/interface.cpp > CMakeFiles/heat_transfer.dir/binding/interface.cpp.i

CMakeFiles/heat_transfer.dir/binding/interface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/heat_transfer.dir/binding/interface.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/binding/interface.cpp -o CMakeFiles/heat_transfer.dir/binding/interface.cpp.s

# Object files for target heat_transfer
heat_transfer_OBJECTS = \
"CMakeFiles/heat_transfer.dir/binding/interface.cpp.o"

# External object files for target heat_transfer
heat_transfer_EXTERNAL_OBJECTS =

heat_transfer.cpython-38-x86_64-linux-gnu.so: CMakeFiles/heat_transfer.dir/binding/interface.cpp.o
heat_transfer.cpython-38-x86_64-linux-gnu.so: CMakeFiles/heat_transfer.dir/build.make
heat_transfer.cpython-38-x86_64-linux-gnu.so: libcore.a
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/local/cuda-10.0/lib64/libcudart_static.a
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/librt.so
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/local/cuda-10.0/lib64/libcudart_static.a
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/librt.so
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/local/cuda-10.0/lib64/libcusparse.so
heat_transfer.cpython-38-x86_64-linux-gnu.so: /usr/local/cuda-10.0/lib64/libcublas.so
heat_transfer.cpython-38-x86_64-linux-gnu.so: CMakeFiles/heat_transfer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module heat_transfer.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/heat_transfer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/heat_transfer.dir/build: heat_transfer.cpython-38-x86_64-linux-gnu.so

.PHONY : CMakeFiles/heat_transfer.dir/build

CMakeFiles/heat_transfer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/heat_transfer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/heat_transfer.dir/clean

CMakeFiles/heat_transfer.dir/depend:
	cd /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/build /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/build /media/qianqianchai/hdiskE/WiSe2021/CSE/HPCAA/Exercise/03/sparse-matvec/build/CMakeFiles/heat_transfer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/heat_transfer.dir/depend

