Prequisites:
    A working python 3.6 installation (python 2 probably also works) with numpy.


Notes on input file:
    The amount of spaces between columns of the input data is inconsistent. We assume
    that there are at least 3 spaces between the input vector and the desired output vector.


Notes on the learning Method:
    The RBF network uses cummulative learning. That is, the weights are only updated once
    all patterns of the training data have been processed.


Notes on the initialization of the RBF centers:
    To initialize the RBF centers a heuristic on the input data (i.e. an input data driven approach)
    is used. The center placements are set to a random subset of size K of the training points and the
    center sizes are 1/K times the diagonal length of the input space.


Running:
    The module can be run using
        python RBF.py
    this will run a short demonstration of the implemented functionality.


Documentation:
    More documentation can be found within RBF.py itself.


Plotting
    When running the demo, all intermediate errors are printed to ./learning.curve.
    The formatting is very straightforward: one real number per line indicating the error after
    performing each training step. A plot can be made with GNUplot, and any labeling
    can be added as desired.