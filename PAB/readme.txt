Prequisites:
    A working python 2.7 installation (python 3 probably also works) with numpy.


Note on input file:
    The amount of spaces between columns of the input data is inconsistent. We assume
    that there are at least 3 spaces between the input vector and the desired output vector.


Running:
    The module can be run using
        python Perzeptron.py
    this will run a short demonstration of the implemented functionality.

    For more advanced use, run python in the main directory and import the module
        import Perzeptron

    Now one can create a perzeptron object and view its parameters by
        p = Perzeptron.Perzeptron([N1, N2, N3, ...])
        print p
    where Ni are positive integers indicating the amount of neurons in each layer layer.
    
    One can evaluate the perzeptron using
        p(X)
    where X should be a numpy array (vector) of the right dimension (Nx1).

    One can set the transfer function and learning rate for each layer using
        p.set_transfer(transfer, i)
        p.set_learning_factor(transfer, i)
    See Util.py for more information on transfer functions.

    To train the perzeptron, use
        p.train(my_patterns)
    where my_patterns should either be a list of pattern objects, which can be
    constructed using
        Perzeptron.Pattern(input_vector, desired_output_vector),
    where the dimensions of input_vector and desired_output_vector should match
    those of p, or the path of a file containing patterns of the right dimension,
    formatted as described in the comment above the parse_training_file function.

    Finally, one can have the average and maximum quadratic error printed for
    a set of verification data using
        p.verify(my_patterns)
    where my_patterns should be as in p.train()


Demo:
    The demo that is executed using 
        python Perzeptron.py
    will take training data from ./training.dat and verification data from ./test.dat
    To use other files please change the values directly within the code.


Documentation:
    More documentation can be found within Perzeptron.py itself.


Plotting
    When running the demo, all intermediate errors are printed to ./learning.curve.
    The formatting is very straightforward: one real number per line indicating the error after
    performing each training step. A plot can be made with GNUplot, and any labeling
    can be added as desired.