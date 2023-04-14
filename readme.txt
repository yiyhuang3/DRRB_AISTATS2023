Additional supplements to the main paper can be found in the file "Appendix.pdf".

The codes for replicating IHDP, Twins, and Credit experiments are packed in the file "code for DRRB".

For IHDP experiments:
The 1000 IHDP datasets can be downloaded from https://www.fredjo.com/
Please open config.txt and modify relevant parameter flags: 'outdir', 'datadir'. Then remain other parameters the same and directly run param_search.py.
After the python process stops, you can open evaluate.py and directly run it. The final results will be saved in the file "result".

For Twins experiments:
The original Twins dataset can be downloaded from https://github.com/AMLab Amsterdam/CEVAE/tree/master/datasets/TWINS. 
100 Twins dataset is generated using the original Twins dataset, and the specific data generating process is the same as
https://github.com/jsyoon0823/GANITE/blob/master/data_loading.py.
Please open config.txt and modify relevant parameter flags to remain the parameters the same as in our paper. 
Then do the same as IHDP experiments to reproduce the results.

For Credit experiments:
The 100 Credit datasets are saved in the file "Credit dataset". The number xxx in "credit_xxx_1-100.test.npz" means the removal ratio. 
Please open config.txt and modify relevant parameters as used in our paper. Then do the same as IHDP experiments to reproduce the results.
