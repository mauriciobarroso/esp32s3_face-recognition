cmake_minimum_required(VERSION 3.5)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(facereg)

target_add_binary_data(${CMAKE_PROJECT_NAME}.elf "certs/ca.pem" TEXT)
target_add_binary_data(${CMAKE_PROJECT_NAME}.elf "certs/certificate.pem.crt" TEXT)
target_add_binary_data(${CMAKE_PROJECT_NAME}.elf "certs/private.pem.key" TEXT)