### 1. Introduction
This repository is the implementation of 
[Structure-Unified M-Tree Coding Solver for MathWord Problem](), which is accepted by EMNLP 2022.

#### Requirements
```python
pip install -r requirements.txt
```

### 2. Usage for SUMC-Solver-PLM
#### train or test SUMC-Solver on Math23k
```
go to ./SUMC_PLM/Math23K
python run_train_solver.py
or 
python run_train_solver.py
```

#### train or test SUMC-Solver on MAWPS
```
go to ./data_processing/
python data_process_mawps.py

go to ./SUMC_PLM/MAWPS
python run_train_mawps_fold_0.py
python run_train_mawps_fold_1.py
python run_train_mawps_fold_2.py
python run_train_mawps_fold_3.py
python run_train_mawps_fold_4.py
or 
python run_test_mwp_solver_bert.py
```

### 3. Usage for SUMC-Solver-RNN
#### train or test SUMC-Solver on Math23k
```
go to ./SUMC_RNN/Math23K_RNN
python run_train_SUMC_Solver.py
or 
python run_test_SUMC_Solver.py
```

#### train or test SUMC-Solver on MAWPS via 5-fold test
```
go to ./data_processing/
python data_process_mawps.py

go to ./SUMC_RNN/MAWPS_RNN
python run_train_mawps_5_folds.py
or 
python run_test_mawps_5_folds.py
```