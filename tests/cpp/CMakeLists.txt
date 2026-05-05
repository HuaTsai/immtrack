include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

function(immtrack_add_test name)
    add_executable(${name} ${name}.cc)
    target_link_libraries(${name} PRIVATE
        immtrack_core
        Catch2::Catch2WithMain
    )
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    )
    catch_discover_tests(${name})
endfunction()

# Tests are added in subsequent tasks. The list grows as features land.
immtrack_add_test(test_angle)
immtrack_add_test(test_sigma_points)
immtrack_add_test(test_traits)
immtrack_add_test(test_ukf_predict)
immtrack_add_test(test_ukf_update)
immtrack_add_test(test_ukf_predict_measurement)
immtrack_add_test(test_hungarian)
immtrack_add_test(test_mahalanobis_cost)
immtrack_add_test(test_track_lifecycle)
immtrack_add_test(test_bbox_tracker)
immtrack_add_test(test_iou3d)
immtrack_add_test(test_amota)
immtrack_add_test(test_concepts)
