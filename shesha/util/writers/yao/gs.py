
def write_gs(file_name, zero_point, lgs_return_per_watt, zenith_angle):
    """Write (append) guide stars parameters to file for YAO

    Args:
        file_name : (str) : name of the file to append the parameter to

        zero_point : (float) : flux for magnitude 0 (ph/m²/s)

        lgs_return_per_watt : (float) : return per watt factor (ph/cm²/s/W)

        zenith_angle : (float) : zenithal angle (degree)
    """
    f=open(file_name,"a+")
    f.write("\n\n//------------------------------")
    f.write("\n//GS parameters")
    f.write("\n//------------------------------")

    # N.B. YAO zeropoint is in flux/telescope aperture, so this value is scaled
    # in outer function
    f.write("\ngs.zeropoint         = " + str(zero_point)+";")
    f.write("\ngs.lgsreturnperwatt  = " + str(lgs_return_per_watt)+";")
    f.write("\ngs.zenithangle       = " + str(zenith_angle) + ";")
