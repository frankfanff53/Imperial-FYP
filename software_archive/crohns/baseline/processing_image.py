import numpy as np
import SimpleITK as sitk


def crop_image(img, index, size):
    """Crop the image.

    Args:
        img: the image to crop.
        index: the top left corner of the bounding box.
        size: the size of the bounding box.

    Returns:
        the cropped image.
    """
    return sitk.RegionOfInterest(img, size, index)


def apply_n4_correction(
    img,
    max_iter,
    control_points,
    histogram_bins=100,
    spline_order=3,
    convergence_threshold=1e-3,
    bias_field_full_width_at_half_maximum=0.1,
    wiener_filter_noise=0.01,
):
    """Apply N4 bias field correction to the image.

    Args:
        img: the image to apply N4 bias field correction to.
        max_iter: the maximum number of iterations applied at each level.
        control_points: the number of control points used in the B-spline
            approximation of the bias field.
        histogram_bins: the number of histogram bins used in the log input
            image intensity histogram. Default is 100.
        spline_order: the spline order used in the bias field estimate. Default
            is 3.
        convergence_threshold: the convergence threshold. Default is 1e-3.
        bias_field_full_width_at_half_maximum: the bias field full width at half
            maximum for the Gaussian deconvolution. Default is 0.1.
        wiener_filter_noise: the wiener filter noise used in the Winer filter.
            Default is 0.01.

    Returns:
        the bias field corrected image.
    """
    cast_image_filter = sitk.CastImageFilter()

    # cast the pixel type to float32
    cast_image_filter.SetOutputPixelType(sitk.sitkFloat32)
    img_cast = cast_image_filter.Execute(img)

    # Apply N4 bias field correction to the image
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(
        (max_iter, max_iter, max_iter, max_iter)
    )
    corrector.SetConvergenceThreshold(convergence_threshold)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(
        bias_field_full_width_at_half_maximum
    )
    corrector.SetWienerFilterNoise(wiener_filter_noise)
    corrector.SetNumberOfHistogramBins(histogram_bins)
    corrector.SetNumberOfControlPoints(
        (control_points, control_points, control_points)
    )
    corrector.SetSplineOrder(spline_order)
    return corrector.Execute(img_cast)


def apply_denoising(img, time_step, iterations):
    """Apply curvature driven flow denoising to the image.

    Args:
        img: the image to apply denoising to.
        time_step: the time step used in the curvature driven flow.
        iterations: the number of iterations used in the curvature
            driven flow.

    Returns:
        the denoised image.
    """
    denoising_filter = sitk.CurvatureFlowImageFilter()
    denoising_filter.SetTimeStep(time_step)
    denoising_filter.SetNumberOfIterations(iterations)
    return denoising_filter.Execute(img)


def apply_laplacian_sharpening(img):
    """Apply Laplacian sharpening to the image.

    Args:
        img: the image to apply Laplacian sharpening to.

    Returns:
        the sharpened image.
    """
    sharpening_filter = sitk.LaplacianSharpeningImageFilter()
    return sharpening_filter.Execute(img)


def apply_slic(
    processed_image,
    max_iter,
    super_grid_size,
    spatial_proximity_weight=5.0,
    enforce_connectivity=True,
    initialization_perturbation=True,
):
    """Apply SLIC superpixel segmentation to the image.

    Args:
        processed_image: the image to apply SLIC superpixel segmentation to.
        max_iter: the maximum number of iterations.
        super_grid_size: the super grid size.
        spatial_proximity_weight: the spatial proximity weight. Default is 5.0.
        enforce_connectivity: whether to enforce connectivity. Default is True.
        initialization_perturbation: whether to use initialization perturbation.
            Default is True.

    Returns:
        the SLIC-segmented image.
    """
    slic_filter = sitk.SLICImageFilter()
    slic_filter.SetMaximumNumberOfIterations(max_iter)
    slic_filter.SetSuperGridSize(
        (super_grid_size, super_grid_size, super_grid_size)
    )
    slic_filter.SetSpatialProximityWeight(spatial_proximity_weight)
    slic_filter.SetEnforceConnectivity(enforce_connectivity)
    slic_filter.SetInitializationPerturbation(initialization_perturbation)
    return slic_filter.Execute(processed_image)


def apply_voting_binary_hole_filling(
    seg,
    radius,
    max_iter,
    majority_threshold,
    background_value,
    foreground_value,
):
    """Apply voting binary hole filling to the image.

    Args:
        seg: the image to apply voting binary hole filling to.
        radius: the radius of the hole filling filter.
        max_iter: the maximum number of iterations.
        majority_threshold: the majority threshold.
        background_value: the background value.
        foreground_value: the foreground value.

    Returns:
        the hole-filled image.
    """
    hole_filling_filter = sitk.VotingBinaryIterativeHoleFillingImageFilter()
    hole_filling_filter.SetBackgroundValue(background_value)
    hole_filling_filter.SetForegroundValue(foreground_value)
    hole_filling_filter.SetRadius((radius, radius, radius))
    hole_filling_filter.SetMaximumNumberOfIterations(max_iter)
    hole_filling_filter.SetMajorityThreshold(majority_threshold)
    return hole_filling_filter.Execute(seg)


def apply_morphological_hole_closing(
    seg,
    kernel_radius,
    kernel_type,
    safe_border=True,
) -> sitk.Image:
    """Apply morphological hole closing to the image.

    Args:
        seg: the image to apply morphological hole closing to.
        kernel_radius: the radius of the kernel.
        kernel_type: the type of the kernel.
        safe_border: whether to use a safe border. Default is True.

    Returns:
        the hole-closed image.
    """

    morph_closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    morph_closing_filter.SetSafeBorder(safe_border)
    morph_closing_filter.SetKernelRadius(
        (kernel_radius, kernel_radius, kernel_radius)
    )
    morph_closing_filter.SetKernelType(kernel_type)
    return morph_closing_filter.Execute(seg)


def get_slic_segments_on_centerline(seg, points):
    """Get the SLIC segments on the centerline.

    Args:
        seg: the SLIC-segmented image.
        points: the points on the centerline.

    Returns:
        the SLIC segments on the centerline.
    """

    def apply_intensity_mask(arr, intensity_mask):
        arr[intensity_mask] = 0

    def generate_intensity_mask(arr, required_intensities):
        return ~np.isin(arr, required_intensities)

    required_intensities = []
    arr = sitk.GetArrayFromImage(seg)

    for point in points:
        x, y, z = point
        # numpy requires (z, y, x) form
        intensity = arr[int(z), int(y), int(x)]
        if intensity not in required_intensities:
            required_intensities.append(intensity)

    intensity_mask = generate_intensity_mask(arr, required_intensities)
    apply_intensity_mask(arr, intensity_mask)

    return sitk.GetImageFromArray(arr)
