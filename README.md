PeakMatcher Usage

Depenedencies

pandas
pytorch
numpy
scipy
matplotlib
pyro-ppl
KDEpy
scikit-learn


Script Description

PeakMatcher.py

This is the peak matching engine. It reads in a reference and target peak lists and tries to find the matching peak for each reference peak in the target. 

USAGE:
      This is the list of flags you must pass
      --reference_peak_list filename  (Path to the reference peak list, sparky formated)
      --target_peak_list filename (Path to the target peak list, sparky formated)
      --reference_cs_columns col1 col2 ...   (a list of chemical shift columns to use for matching e.g. w1 w2)
      --target_cs_columns col1 col2 ... ( a list of chemical shift columsn to use for matching e.g. w1 w2, order must correspond to that to the reference cs name)
      --reference_peak_list_error  error1 error2  (a list of errors for the corresponding reference columns (e.g. 0.015 ppm for N and 0.0015 for H)
      --target_peak_list_error error1 error2 ( a list of errors for the corresponding target columns (e.g. 0.015 ppm for N and 0.0015 for H)
      --output_directory path (directory for the outputed files)
      Regularization Terms:
      The following are regularzation terms, they all have default values that are generally applicable to most situations
      --expected_fraction_csp 0.05 (The fraction of reference peaks that you expect to be perturbed )
      --variance_scale_fraction_csp 5.0 (Scaling factor for the variance of the prior distribution of fraction_csp variance=scale*expected_fraction_csp^2)
      --expected_fraction_missing 0.02 (The fraction of reference peaks you expect to missing)
      --variance_scale_fraction_missing 1.0 (Scaling factor for the variance of the prior distribution of fraction_missing  variance=scale*expected_fraction_missing^2)
      --gradient_convergence 1E-5 (magnitude threshold for all gradient dimensions to be considered converged)
      --display_distributions (use when you want the distributions to flash on the screen after each EM step)
      --confidence_cutoff 0.90 (minimum matching probability to be considered a confident match)
      --compute_reference_offset (use when you want to consider that there may be a slight reference offset between spectra and would like to consider that offset during matching)


      

      
      
