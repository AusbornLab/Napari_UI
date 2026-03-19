from tifffile import TiffFile

path = r"C:\Users\antho\Downloads\astro 1_Lng 1.tif"   # change this

with TiffFile(path) as tif:
    print("=== Basic info ===")
    print("Series shape:", tif.series[0].shape)
    print("Axes:", tif.series[0].axes)
    print()

    print("=== ImageJ metadata ===")
    ij = tif.imagej_metadata
    print(ij)
    print()

    print("=== OME metadata ===")
    ome = tif.ome_metadata
    print(ome[:2000] if ome else None)   # print first part only
    print()

    page = tif.pages[0]
    tags = page.tags

    print("=== TIFF tags ===")
    for key in ["XResolution", "YResolution", "ResolutionUnit"]:
        if key in tags:
            print(key, "=", tags[key].value)
    print()

    voxel_z = None
    voxel_y = None
    voxel_x = None
    unit = None

    # 1) ImageJ-style metadata
    if ij is not None:
        voxel_z = ij.get("spacing", voxel_z)
        unit = ij.get("unit", unit)

    # 2) TIFF X/Y resolution tags
    if "XResolution" in tags:
        num, den = tags["XResolution"].value
        if num != 0:
            voxel_x = den / num
    if "YResolution" in tags:
        num, den = tags["YResolution"].value
        if num != 0:
            voxel_y = den / num

    if "ResolutionUnit" in tags:
        ru = tags["ResolutionUnit"].value
        print("Raw ResolutionUnit:", ru)

    print("=== Estimated voxel size ===")
    print("Z:", voxel_z, unit)
    print("Y:", voxel_y, unit)
    print("X:", voxel_x, unit)
