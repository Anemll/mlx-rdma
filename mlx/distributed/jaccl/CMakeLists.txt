if(MLX_BUILD_CPU
   AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin"
   AND MACOS_SDK_VERSION GREATER_EQUAL 26.2
   OR MLX_BUILD_JACCL)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jaccl.cpp)
  # On macOS, link against librdma which provides InfiniBand Verbs
  # Use PUBLIC so shared libraries linking to mlx also get rdma dependency
  target_link_libraries(mlx PUBLIC rdma)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_jaccl.cpp)
endif()
