Prequisites:
    A working python 2.7 installation (python 3 probably also works) with numpy.

Running:
    The module can be run using
        python Perzeptron.py
    this will run a short demonstration of the implemented functionality.

    For more advanced use, run python in the main directory and import the module
        import Perzeptron

    Now one can create a perzeptron object and view its parameters by
        p = Perzeptron.perzeptron(my_N, my_M)
        print p
    where my_N, my_M are positive integers.
    
    One can evaluate the perzeptron using
        p(X)
    where X should be a numpy array (vector) of the right dimension (Nx1).    

    To manually set the weights of p, use
        p.set_weights(my_weights)
    where my_weights should either be a numpy array of dimensions (N+1)xM or
    the path of a file in the containing a numpy array of those dimensions 
    in standard formatting (such that it can be read by numpy.loadtxt)

    To train the perzeptron, use
        p.train(my_patterns, x)
    where my_patterns should either be a list of pattern objects, which can be
    constructed using
        Perzeptron.pattern(input_vector, desired_output_vector),
    where the dimensions of input_vector and desired_output_vector should match
    those of p, or the path of a file containing patterns of the right dimension,
    formatted as described in the comment above the parse_training_file function.
    The second argument x is used to specify how often p should be trained.

    Finally, one can verify that the perzeptron has succesfully learned the 
    training data using
        p.verify(my_patterns)
    where my_patterns is as above.

Documentation:
    More documentation can be found within Perzeptron.py itself.