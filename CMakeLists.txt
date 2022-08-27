cmake_minimum_required(VERSION 3.13.4)
project(setFTs)

add_subdirectory(pybind11)
pybind11_add_module(fast src/fast/fast.cpp)