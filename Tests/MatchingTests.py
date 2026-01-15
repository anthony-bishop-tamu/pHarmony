import sys, subprocess, shlex, textwrap
import importlib_resources
from pHarmony.PeakMatcher import main as pm_main


data_directory =  importlib_resources.files('Tests.TestData')

#case1

reference_list = data_directory/'case1'/'IL1B_FragmentScreen_Reference_1.list'
target_list = data_directory/'case1'/'IL1B_FragmentScreen_Fragment_232.list'
output_directory = data_directory/'case1'/"test_output"
arguments = [ "--reference_peak_list", str(reference_list),
              "--reference_cs_column_names", 'w1','w2',
              "--reference_peak_list_error", "0.015", "0.0015",
              "--target_peak_list", str(target_list),
              "--target_cs_column_names", "w1","w2",
              "--target_peak_list_error", "0.015", "0.0015",
              "--output_directory",str(output_directory),
              "--expected_fraction_csp", "0.1",
              "--variance_scale_fraction_csp", "1",
              "--expected_max_csp", "0.1",
              "--gradient_convergence", "1E-5",
              "--confidence_cutoff", "0.9",
              "--CSP_scaling_factor", "0.101", "1",
              "--log_file"]

#pm_main(arguments)

#case2
reference_list = data_directory/'case2'/'IL1B_manually_transferred_assignments_fitted_81'
target_list = data_directory/'case2'/'fitted_401.list'
output_directory = data_directory/'case2'/"test_output"
arguments = [ "--reference_peak_list", str(reference_list),
              "--reference_cs_column_names", 'w1','w2',
              "--reference_peak_list_error", "0.015", "0.0015",
              "--target_peak_list", str(target_list),
              "--target_cs_column_names", "w1","w2",
              "--target_peak_list_error", "0.015", "0.0015",
              "--output_directory",str(output_directory),
              "--expected_fraction_csp", "0.1",
              "--variance_scale_fraction_csp", "1",
              "--expected_max_csp", "0.1",
              "--gradient_convergence", "1E-5",
              "--confidence_cutoff", "0.9",
              "--CSP_scaling_factor", "0.101", "1",
              "--log_file"]

#pm_main(arguments)

#case3
reference_list = data_directory/'case3'/'km_processingAndPeakPicking_2DTROSY_1031.list'
target_list = data_directory/'case3'/'km_processingAndPeakPicking_2DTROSY_1041.list'
output_directory = data_directory/'case3'/"test_output"

arguments = [ "--reference_peak_list", str(reference_list),
              "--reference_cs_column_names", 'w1','w2',
              "--reference_peak_list_error", "0.015", "0.0015",
              "--target_peak_list", str(target_list),
              "--target_cs_column_names", "w1","w2",
              "--target_peak_list_error", "0.015", "0.0015",
              "--output_directory",str(output_directory),
              "--expected_fraction_csp", "0.1",
              "--variance_scale_fraction_csp", "1",
              "--expected_max_csp", "0.2",
              "--gradient_convergence", "1E-5",
              "--confidence_cutoff", "0.9",
              "--CSP_scaling_factor", "0.101", "1",
              "--log_file"]

#pm_main(arguments)

#case5
reference_list = data_directory/'case5'/'km_processingAndPeakPicking_2DTROSY_1125.list'
target_list = data_directory/'case5'/'km_processingAndPeakPicking_2DTROSY_1001.list'
output_directory = data_directory/'case5'/"test_output"

arguments = [ "--reference_peak_list", str(reference_list),
              "--reference_cs_column_names", 'w1','w2',
              "--reference_peak_list_error", "0.015", "0.0015",
              "--target_peak_list", str(target_list),
              "--target_cs_column_names", "w1","w2",
              "--target_peak_list_error", "0.015", "0.0015",
              "--output_directory",str(output_directory),
              "--expected_fraction_csp", "0.2",
              "--variance_scale_fraction_csp", "1",
              "--expected_max_csp", "0.2",
              "--gradient_convergence", "1E-5",
              "--confidence_cutoff", "0.9",
              "--CSP_scaling_factor", "0.101", "1",
              "--log_file"]

#pm_main(arguments)

#case5
reference_list = data_directory/'case6'/'km_IL1B_PeakMatcherBenchmark_HNCO_1003_filtered.list'
target_list = data_directory/'case6'/'km_IL1B_PeakMatcherBenchmark_HNCO_1073_filtered.list'
output_directory = data_directory/'case6'/"test_output"

arguments = [ "--reference_peak_list", str(reference_list),
              "--reference_cs_column_names", 'w2','w3',
              "--reference_peak_list_error", "0.015", "0.0015",
              "--target_peak_list", str(target_list),
              "--target_cs_column_names", "w2","w3",
              "--target_peak_list_error", "0.015", "0.0015",
              "--output_directory",str(output_directory),
              "--expected_fraction_csp", "1E-12",
              "--variance_scale_fraction_csp", "1",
              "--expected_max_csp", "0.2",
              "--gradient_convergence", "1E-5",
              "--confidence_cutoff", "0.9",
              "--CSP_scaling_factor", "0.101", "1",
              "--log_file"]

#pm_main(arguments)

#case5
reference_list = data_directory/'case7'/'fitted_not_referenced_471.list'
target_list = data_directory/'case7'/'fitted_121.list'
output_directory = data_directory/'case7'/"test_output"

arguments = [ "--reference_peak_list", str(reference_list),
              "--reference_cs_column_names", 'w1','w2',
              "--reference_peak_list_error", "0.015", "0.0015",
              "--target_peak_list", str(target_list),
              "--target_cs_column_names", "w1","w2",
              "--target_peak_list_error", "0.015", "0.0015",
              "--output_directory",str(output_directory),
              "--expected_fraction_csp", "0.1",
              "--variance_scale_fraction_csp", "3",
              "--expected_max_csp", "0.1",
              "--gradient_convergence", "1E-5",
              "--confidence_cutoff", "0.9",
              "--CSP_scaling_factor", "0.101", "1",
              "--log_file"]

pm_main(arguments)