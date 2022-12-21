cmake_minimum_required(VERSION 2.8.12)
project(MoonmatchData)

file(DOWNLOAD
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp
    data/de421.bsp
    EXPECTED_HASH SHA256=08b20db2ae22488650641c5a9033e5bfda4b1c4b440cfeaf20f621cfa18ecdb3
    SHOW_PROGRESS
)
file(DOWNLOAD
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls
    data/latest_leapseconds.tls
    # Leap seconds could change any year so don't check hash
    SHOW_PROGRESS
)
file(DOWNLOAD
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc
    data/moon_pa_de421_1900_2050.bpc
    EXPECTED_HASH SHA256=656f90616403d75a75f0cd6c8830fc5b44f8cb4facb5ccb8915e752b397520cf
    SHOW_PROGRESS
)
file(DOWNLOAD
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf
    data/moon_080317.tf
    EXPECTED_HASH SHA256=78732477b96f9863e7b0d65bcee3c22b8707ca5ed0db56d1173319cb2e8c7993
    SHOW_PROGRESS
)

file(DOWNLOAD
    https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/CHANGE3/NAC_DTM_CHANGE3.TIF
    map/NAC_DTM_CHANGE3.TIF
    EXPECTED_HASH SHA256=506cc3397f551f571fdc63cf761abedb94ceaacdddd5d245317e11bde0381504
    SHOW_PROGRESS
)

add_custom_target(
    dummy ALL
)
add_custom_command(
    TARGET dummy
    COMMAND gdaldem hillshade ${CMAKE_CURRENT_SOURCE_DIR}/map/NAC_DTM_CHANGE3.TIF ${CMAKE_CURRENT_SOURCE_DIR}/map/NAC_DTM_CHANGE3_SHADE2.TIF -of GTiff -compute_edges -alg ZevenbergenThorne -az 110 -alt 30
)
