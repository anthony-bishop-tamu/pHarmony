from src.PeakMatcher.PeakMatcher import standalone_match_peaks
import src.PeakMatcher
import importlib_resources
import logging



data_directory =  importlib_resources.files('Tests.TestData')

#case1
reference_list = data_directory/'case1'/'IL1B_FragmentScreen_Reference_1.list'
target_list = data_directory/'case1'/'IL1B_FragmentScreen_Fragment_232.list'
output_directory = data_directory/'case1'/"test_output"

standalone_match_peaks(reference_list,['w1','w2'], [0.015,0.0015],
                        target_list,['w1','w2'], [ 0.015,0.0015],
                        output_directory,
                        0.1,0.5,
                        0.1,2.0,
                        0.1,
                        1E-5,
                        True,False,0.9, [0.101,1],log_level=20,log_file=True
                         )

#case2
reference_list = data_directory/'case2'/'IL1B_manually_transferred_assignments_fitted_81'
target_list = data_directory/'case2'/'fitted_401.list'
output_directory = data_directory/'case2'/"test_output"

standalone_match_peaks(reference_list,['w1','w2'], [0.03,0.003],
                        target_list,['w1','w2'], [ 0.03,0.003],
                        output_directory,
                        0.1,0.5,
                        0.1,2.0,
                        0.1,
                        1E-5,
                        False,False,0.9, [0.101,1], log_level=20,log_file=True)