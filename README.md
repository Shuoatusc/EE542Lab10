# EE542Lab10
1.
Create a python3 virtual environment 
virtualenv -p python3 envname
source envname/bin/activate

2. install all the packages needed. (sklearn, pandas)

pip install sklearn 
pip install pandas
Pip install matplotlib

3. Source codes:
check.py:
		 This is to check the integrity for the downloaded miRNA files
		 python check.py.  

parse_file_case_id.py:  
		This is to get the unique file id and the corresponding case ids.	
		python parse_file_case_id.py

request_meta.py: This is to request the meta data for the files and cases.
		python request_meta.py

gen_miRNA_matrix.py: This is to generate the miRNA matrix and labels for all the files 
		python gen_miRNA_matrix.py

predict.py : This is for applying models to the miRNA matrix for normal/ tumor sample detection.
		python predict.py
predict_sagemaker.py: This is for running in AWS sagemaker
