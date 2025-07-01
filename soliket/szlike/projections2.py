def convolve_Hankel(thta_smooth, prof2D, beam_txt, thta_use, beam_response=False):
    rht = RadialFourierTransform(
        n=200, pad=100, lrange=[170.0, 1.4e6]
    )  # note hard values, n here needs to be same size as rad in project
    profMap = np.interp(rht.r, thta_smooth, prof2D)
    lprofs = rht.real2harm(profMap)
    lprofs *= f_beam_fft(beam_txt, rht.ell)

    if beam_response is not False:
        respTF = f_response(beam_response, rht.ell)
        # multiply by the response
        lprofs *= respTF

    rprofs = rht.harm2real(lprofs)
    # padding
    r_unpad, rprofs = rht.unpad(rht.r, rprofs)
    prof2D_beam = interp1d(
        r_unpad.flatten(),
        rprofs.flatten(),
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )(thta_use)
    return prof2D_beam




"""Can be done once before the profile"""
# n must be same size as l array
# 
rht = RadialFourierTransform(n=200, pad=100, lrange=[170.0, 1.4e6]


def project_tsz(Pths, rs, thetas, z,
                outer_ring = np.sqrt(2),
                f_smooth = 3.5,
                N_theta = 100):


    
    theta_smooth = np.arange(1, N_theta*f_smooth+1) * dtht / resolution_factor

    # Interpolate the 3D pressure profile to the radii values of the integration
    Pth_inter = interp1d(rs, Pths, bounds_error=False, fill_value=0.0)


    