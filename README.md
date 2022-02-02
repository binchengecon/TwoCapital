# TwoCapital
Repo for R&amp;D model

## Prerequisite

- PETSc installed with flag `--with-petsc4py --with-debugging=no PETSC_ARCH=arch-xx-opt`
- Maintain PETSc, `export PETSC_ARCH=arch-xx-opt`, 
    `export PETSC_DIR=<your petsc file>`
    `export PYTHONPATH=$PETSC_DIR/arch-xx-opt/lib`
- `pip install -r requirements.txt`
- To use C implementation `pip install ./src/linearsystemcore`
- (Optional) installation of `SolveLinSys` package `pip install ./src/cppcore`

## Scripts and Model

A write-up is under the `./write-ups/write-up.pdf`

In py file, `linearsolver` stands for the linear solver to use: 
available solution:

- `pestsc` for PETSc + C implementation of coefficient matrix
- `petsc4py` for PETSc and numpy sparse matrix
- `eigen` for `SolveLinSys`


`post-jump-test.py` corresponds to section 1.1, with normalized capitals

`post-jump-change.py` corresponds to section 1.2.2 with state variables log K, R, Y



Errors

<img src="./fc-err.png" />
