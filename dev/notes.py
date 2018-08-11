
# NOTE: some stuff can be moved to the tb_numpy.
#

# Implementation requests
# * tb_augmentation.py
# * tb_data.py
# * tb_debugging.py
# * tb_experiments.py
# * tb_filesystem.py
# * tb_io.py
# * tb_logging.py
# * tb_plotting.py
#     * Add grid plots.
#     * Add histograms (1D, 2D).
#     * Add scatter plots (1D, 2D, 3D).
# * tb_preprocessing.py
# * tb_project.py
# * tb_random.py
# * tb_remote.py
# * tb_resource.py
# * tb_training.py
#     * Model checkpoint and resuming.

# * tb_tensorflow.py
# * tb_metrics.py
# * tb_pytorch.py
# * tb_serving.py
# * tb_deep_architect.py
# * tb_numpy (?)

# Additional aspects to contemplate.

# tb_tensorflow and tb_pytorch (I'm not sure what to do here; I think that
# certain things are probably useful to have such as what is the best way of
# defining new models and perhaps some class definition.)

# Design principles:
# * Working with lists most of the time
# * d stands for dictionary.
# * Suggests the conceptualization of recurring transformations that would otherwise be spread out throughout the code.
# * Keep it simple and extensible.
# * Tools are overly general; we provide a simplification.
# * It should be natural for the person using the functionality.
# * Translation layer from more flexible language to a more simple language.
# * Collection of utility scripts rather than a big system with a lot of interactions.

# tb_features.py
# * Implement the different combinators for features.

# Other potential aspects to consider
# * Model serving.
# * Email notifications when job or process terminates.
# * Tools to work with a server in a more interactive way.

# Design principle: loosely decoupled models.

# general implementation:
# * Python 3 compatibility
# * Additional tests for functionality.

# add more plot objects that are easy to use for different configurations.

# Going about making this more useful to you
# * Understand the format to run the code.
# * Understand the creation of examples.
# * Understand running on the server.
# * Adapt the examples to run on the server to suit your needs.

# * Simple interfaces for models.
# * Tools for preprocessing and training.
# * Add more system information (GPU that was ran on).
# * Easy data preprocessing.
# * Loading data
# * Working with models
# * Evaluation
# * Visualization of model predictions.
# * Common error checking operations.
# * Operate directly on files.
# * Easy to use tensorboard for simple things.
# * Get all files matching a pattern in a directory.
# * Simple scripts for working with images and videos.
# * Get to the point
# * Improve the loading of experiments and the manipulation of experiments.
# * Creating different hyperparameter configurations.
# * Add common step size strategies.
# * Add more common training logic like step size reduction and what not.
# * Make it trivial to run on a remote machine (these configurations should be easy to set up).
# * Improve plotting functionality.
# * Add functionality for darch with some common models.
# * Make sure that it is easy to process the resulting logs.
# * Inspect the variations among different dictionaries.
# * Command line based tools for working directly from the command line (perhaps some batch file). An example is tb_interact.py run --alias sync_folder_to_matrix (stuff like that).
# * Get available resources from the command line.
# * Add some simple hyperparameter tuning.
# * Add some simple graph visualization.
# * Working easily on remote paths.
# * Download all files of a given type from a webpage.
# * Support for JSON formatting?
# * Working easily with ordered dicts.
# * Profiling and debugging hooks.
# * Working with numpy for the computational interface of the jobs.
# * Combining a few of these files.
# * Better tools for managing directories.
# * Write down typical use cases where it would be appropriate to extend the functionality.
# * Make it really easy to run a command like it was on the server.
# * Ability to easily extend (in a sane way) the models that are in there.
# * Better tools for featurization.
# * Support for nested experiments.
# * Exploration of CSV files can be done through pandas.
# * Add some support to easily work with an hyperparameter optimization toolbox.
# * Functionality to inspect the mistakes made by a model.
# * Run directly from a config file.
# * Add creation of Tikz figures in python.
# * Logging is mostly done through JSON.
# * Adding your own interaction commands.
# * Print available commands in interact.
# * Run on a SLURM managed clustered with all the dependencies.
# * Plot quantiles easily.
# * Map a file line by line somehow (e.g., by calling some function on it).
# * Scripts for managing environments and configurations. These can be done in Python, e.g., setting the appropriate environment variables.
# * Simple processing of output from bash commands. Going seamlessly between bash commands and Python processing. Should be able to pipe to a Python command.
# * Functionality to help write reentrant code.
# * Run configurations to reduce the amount of repetition that is needed when running things.
# * Running any command of the toolbox via tb_interact. Useful for file manipulation.
# * Fix the randomness for Tensorflow and Pytorch and potentially other code.
# * Add test for the currently implemented functionality.
# * Easy to use functionality to easily generate processing folders.
# * For the interact, I also want to generate code that is bash instead of Python. Does this work for function calls.
# * Dynamic definition of new function calls. Potentially by combining with some simple iteration.
# * Tools for data collection (this just means constructing a JSON incrementally, I suppose), potentially with some more processing information.
# * More standard loggers to make sure that it can work easily with different models.
# * Some common tools for crawling and such. Download all pdfs from a webpage.
# * Create a tb_web_crawling.py for downloading stuff from the web; this is similar to going over folder in some sense.
# * Very simple to create a simple dataset by crawling data from the web.
# * Packaging data in some of these simple formats.
# * Common readers and writers for some simple data.
# * Model serving based on the type of the data.
# * Keeping running configs in a JSON is nice. fast, noop, default.
# * Easy processing of text log files, either via matching. Something that would be interesting is to combine mapping over files with something better. A lot of it is just iteration over files.
# * How to plot sequences directly from the command line, and how can this be done.
# * Ability to very easily interface with the C code. What are examples of C functionality that would be better ran in C.
# * APIs for training and serving.
# * Easy download of data and what not.
# * Including the toolbox as a submodule, or copying the needed files.
# * Install a specific version locally.
# * Easy setup of the environment to run on certain types of data.
# * Clear a latex file from comments.
# * Nice interfaces for data.
# * Define some general formats to deal with data.
# * Very simple way of defining simple trainers and hyperparameters.
# * Managing dependencies.
# * Addressing some of the Pytorch pitfalls
# * Easy masking and masking working for different functionality of the model.
# * Easy definition of a conv net and LSTM and some training code.
# * Functionality to make the server as a seamless extension of the local machine.
# * Easy logging emails with information about how the results are going. This can be useful to deal with the model.
# * Use the inspect module for some code manipulation.
# * Some automatic experiment generation for running certain types of experiments automatically.
# * Create a simple repository of some Pytorch and Tensorflow models.
# * Simple language independent training schedules.
# * Tools for model inspection.
# * Smarter tools for running configuration folders and to process their results.
# * Creation of composite tools.
# * Easy to create videos and simple animations with images.
# * Add better naming conventions for running different experiments.
# * Treating the experiment folder nicely.
# * Registering commands locally should help run things very easily, like a sequence
# of commands to run.
# * Working with tree dictionaries.
# * Conditional logging information.
# * Add functionality to run periodically.
# * Have a config manager to write experiments easily.
# * Improve the run on a SLURM managed cluster such that it is easy to adapt to a new one.
# * Pretrained models.
# * Tools for helping generating all the results of a paper based on a single script.
# * Easy to stress test a model and make sure that it works.
# * Dataset manager that allows to download the data in disk.
# * Write a mini-scheduler for non-managed node servers.
# * List files in a remote folder.
# * Easy to apply regular expressions and substitutions.
# * Send a downloaded version of the experiments upon completion.
# * Design some simple instructions/procedure to get things set up in a server.
# * Design some simple procedure to get a Docker container running with the experiments.
# * Setting up ssh id without a server. (create a new key, )
# * Download files of a certain type from a webpage.
# * Add option to run command locally: run_on_local
# * Add abort if it exists to the file checks.
# * Add a function to draw the dependencies of a project, assuming that there is a root from which files are called.
# * Add functionality to see how much time has been waiting on the queue.
# * Functionality to know my jobs in the queue.
# * Simplify the run on lithium node. it still has too many options for most cases.
# * Adding a function to extract from the toolbox is convenient. Check how to do this. This is convenient when we don't want to depend on the whole toolbox..
# * Add functionality to make it easy to work with the assyncronous computation.
# * Add scripts to periodically clean the folder. It would be nice to add one of these scripts in Python
# * Make sure that I can specify the axis of the plotter to be easier to use (like log vs linear).
# * Add some functionality to make sure that I can easily substitute all occurrences of strings in a code.
# * Add functionality to take the top elements of a vector

# it should be easy to take a folder and run it on the cluster with some
# configuration, then it should be easy to get the results back, maybe waiting
# or not for it to finish.

# what is a good description of a featurizer. I think that the definition of
# the data is relatively stable, but featurizers are malleable, and depend
# on design choices. the rest is mostly independent from design choices.

# binary search like featurization. think about featurization for
# directory like structures.
# this is binarization. check how they do it in scikit-learn.

# download certain files periodically or when they terminates. this allows to
# give insight into what is happening locally.
# there is syncing functionality

# make it easy to use the rsync command.

# managing jobs after they are running in the server.

# TODO: write a file with information about the system that did the run.
# gathering information about the system is important

# TODO: and perhaps kill some of the jobs, or have some other file
# that say ongoing. define what is reasonable to keep there.

# why would you change an experiment configuration. only if it was due to
# bug. what if the code changes, well, you have to change the other experiments
# according to make sure that you can still run the exact same code, or maybe
# not.

# TODO: perhaps it is possible to run the model in such a way that makes
# it easy to wait for the end of the process, and then just gets the results.
# for example, it puts it on the server, sends the command, and waits for
# it to terminate. gets the data every time.

# TODO: the question is which are the cpus that are free.
# NOTE: it is a good idea to think about one ssh call as being one remote
# function call.

# ideally, you just want to run things once.
# also, perhaps, I would like to just point at the folder and have it work.

# TODO: develop functions to look at the most recent experiments, or the files
# that were change most recently.

# the commands are only pushed to the server for execution.

# easy to do subparsers and stuff, as we can always do something to
# put the subparsers in the same state are the others.
# can use these effectively.

# TODO: add something to run periodically. for example, to query the server
# or to sync folders. this would be useful.

# TODO: it is possible to have something that sync folders
# between two remote hosts, but it would require using the
# current computer for that.

# add error checking

# TODO: check output or make sure that things are working the correct way.
# it may be preferable to do things through ssh sometimes, rather than
# use some of the subprocess with ssh.

# NOTE: it is important to be on the lookout for some problems with the
# current form of the model. it may happen that things are not properly
# between brackets. this part is important

# there is questions about being interactive querying what there is still
# to do there.

# there is only part of the model that it is no up. this is better.

# there is more stuff that can go into the toolbox, but for now, I think that
# this is sufficient to do what I want to do.

# add stuff for error analysis.

# TODO: functionality for periodically running some function.

# TODO: more functionality to deal with the experiment folders.

# for reading and writing CSV files, it is interesting to consider the existing
# CSV functionality in python

# look at interfacing nicely with pandas for some dataframe preprocessing.

# NOTE: neat tools for packing and unpacking are needed. this is necessary
# to handle this information easily.

# TODO: mapping the experiment folder can perhaps be done differently as this is
# not very interesting.

# dealing with multiple dictionaries without merging them.

# going over something and creating a list out of it through function
# calls, I think that is the best way of going.

# NOTE: a list is like a nested dictionary with indices, so it
# should work the same way.

# NOTE: for flatten and stuff like that, I can add some extra parts to the model
# that should work nicely, for example, whether it is a list of lists
# or not. that is nicer.

# there are also iterators, can be done directly. this is valid, like
# [][][][][]; returns a tuple of that form. (s, s, ...)
# this should work nicely.

# question about iterator and map,
# I think that based on the iterator, I can do the map, but it is probably
# a bit inefficient. I think that

# NOTE: some of these recursive functions can be done with a recursive map.

# most support for dictionaries and list and nested mixtures of both.
# although, I think that dictionaries are more useful.

# TODO: make it easier to transfer a whole experiment folder to the server and
# execute it right way.

# develop tools for model inspection.

# TODO: some easy interface to regular expressions.

# TODO: stuff to inspect the examples and look at the ones that have the most
# mistakes. this easily done by providing a function

# there is stuff that needs interfacing with running experiments. that is the
# main use of this part of this toolbox.

# think about returning the node and the job id, such that I can kill those
# jobs easily in case of a mistake.

# TODO: add stuff for coupled iteration. this is hard to do currently.
# think about the structure code.

# TODO: work on featurizers. this one should be simple to put together.

# NOTE: it is a question of looking at the number of mistakes of a
# model and see how they are split between types of examples.

# or generate confusion matrices easily, perhaps with bold stuff for the largest
# off-diagonal entries.

# Working with the configs, I think that that is interesting.
# the folders do not matter so much, but I think that it is possible
# do a mirror of a list [i:]

# TODO: add ML models unit tests.
# like running for longer should improve performance.
# performance should be within some part of the other.
# do it in terms of the problems.

# <IMPORTANT> TODO: logging of different quantities for debugging.
# greatly reduce the problem.
# copy the repository somewhere, for example, the debug experiments.

# TODO: these can be improved, this is also information that can be added
# to the dictionary without much effort.

# TODO: function to get a minimal description of the machine in which
# the current model is running on.

# loss debugging. they should go to zero.
# TODO: dumb data for debugging. this is important to check that the model is working correctly.

# question about the debugging. should be a sequence of
# objects with some descriptive string. the object should probably also
# generate a string to print to a file.
# this makes sense and I think that it is possible.

# TODO: for example, if I'm unsure about the correctness of some variable
# it would be nice to register its computation. how to do that.
# I would need to litter the code with it.

# TODO: passing a dictionary around is a good way of registering information
# that you care about. this is not done very explicitly. using just a dictionary
# is not a good way. perhaps reflection about the calling function would be
# a good thing, and some ordering on what elements were called and why.

# TODO: add gradient checking functionality.

# TODO: add some easy way of adding tests to the experiment folders. like
# something as to be true for all experiments.
# also something like, experiments satisfying some property, should
# also satisfy some other property.

# NOTE: bridges is SLURM managed. I assume it is only slightly different.

# do something to easily register conditions to test that the experiments
# should satisfy.

# add a function to say which one should be the empty cell placeholder in the
# table.

# TODO: stuff for error analysis. what are the main mistakes that the model
# is doing. there should exist a simple way of filtering examples by the number
# of mistakes.

# curious how debugging of machine learning systems work. based on performance.
# differential testing.

# have default values for the hyperparameters that you are looking at.
# for example, for step sizes, there should exist a default search range.\

# TODO: do the featurizer and add a few common featurizers for cross product
# and bucketing and stuff.

# think about how to get the featurizer, that can be done either through
# it's creation or it is assumed that the required information is passed as
# argument.

# this is a simple map followed by a sort or simply sorting by the number of
# mistakes or something like that.
# perhaps something more explicit about error analysis.

# generation of synthetic data.

# TODO: think about the implementation of some stateful elements.
# I think that the implementation of beam search in our framework is something
# worthy.

# simple binarization,

# TODO: check utils to train models online. there may exist stuff that
# can be easily used.

# TODO: interface with Pandas to get feature types of something like that
# I think that nonetheless, there is still information that needs to
# be kept.

# tests to make the model fail very rapidly if there are bugs.
# functions to analyze those results very rapidly.

# TODO: stuff for checkpointing, whatever that means.

# add functionality to send email once stops or finishes.
# this helps keep track of stuff.

# TODO: some easy interfacing with Pandas would be nice.
# for example to create some dictionaries and stuff like that.

# very easy to work on some of these problems, for example,
# by having some patterns that we can use. this is interesting in terms of
# model composition, what are typical model composition strategies?
# for example, for models with embeddings.

# TODO: functionality to work with ordered products.

# NOTE: perhaps it is possible to keep scripts out of the main library folder,
# by making a scripts folder. check if this is interesting or not.
# still, the entry point for running this code would be the main folder.
# of course, this can be changed, but perhaps it does not matter too much.

# NOTE: what about stateful feature extraction. that seems a bit more
# tricky.

# TODO: online changes to configurations of the experiment.
# this would require loading all the files, and substituting them by some
# alteration. this is kind of like a map over the experiments folder.
# it does not have to return anything, but it can really, I think.
# returning the prefix to the folder is the right way of doing things.

# make it easy to do the overfit tests.
# it is a matter of passing a lot of fit functions. this can be
# done online or by sampling a bunch of models from some cross product.

# NOTE: some of these aspects can be better done using the structure
# information that we have introduced before.

# TODO: add functionality to make it easy to load models and look at it
# an analyze them.

# TODO: error inspection is something that is important to understand

# some subset of the indices can be ortho while other can be prod.

# TODO: in the config generation, it needs to be done independently for
# each of the models. for example, the same variables may be there
# but the way they are group is different.

# TODO: for the copy update, it is not a matter of just copying
# I can also change some grouping around. for example, by grouping
# some previously ungrouped variables.
# it is possible by rearranging the groups.
# the new grouping overrides the previous arrangement.
# composite parameters that get a list of numbers are interesting, but have
# not been used much yet. it is interesting to see
# how they are managed by the model.

# TODO: add the change learning rate to the library for Pytorch.

# TODO: I have to make sure that the patience counters work the way I expect
# them to when they are reset, like what is the initial value in that case.
# for example, this makes sense in the case where what is the initial
# number that we have to improve upon.
# if None is passed, it has to improve from the best possible.

# for the schedules, it is not just the prev value, it is if it improves on the
# the prev value or not.

# TODO: functionality to rearrange dictionaries, for example, by reshuffling things
# this is actually quite tricky.

# TODO: have an easy way of doing a sequence of transformations to some
# objects. this is actually quite simple. can just just

# TODO: do some form of applying a function over some sequence,
# while having some of the arguments fixed to some values.
# the function needs to have all the other arguments fixed.

# TODO: also make it simple to apply a sequence of transformations
# to the elements of a string.

# TODO: dict updates that returns the same dictionary, such that they
# can be sequenced easily.

# easy to run things conditionally, like only runs a certain function
# if some condition is true, otherwise returns the object unchanged.

# TODO: a trailing call is going to be difficult, as it is not an object in
# general, nesting things, makes it less clear. otherwise, it can
# pass as a sequence of things to execute. looks better.
# conditions.

# NOTE: a bunch of featurizers that take an object and compute something based
# on that object.
# sequence of transformations.

# TODO: code to run a function whenever some condition is verified.
# TODO: code to handle the rates and to change the schedules whenever some
# condition is verified.

# TODO: stuff with glob looks nice.

# TODO: an interesting thing is to have an LSTM compute an embedding for the
# unknown words. I think that this is a good compromise between
# efficiency and generalization.

# NOTE: some auxiliary functions that allows to easily create a set of experiments
# there are also questions.

# each config may have a description attached.

# NOTE: some of the stuff from the plots can come out of current version of
# of the plots.

# the run script for a group of experiments is going to be slighly different
# what can be done there. it would be inconvenient for large numbers to send
# it one by one. I guess it it possible to easily do it, so not too much of a
# problem.

# TODO: ortho increments to a current model that may yield improvements.
# or just better tools to decide on what model to check.

# I think that it is going to be important to manage the standard ways of
# finding good hyperparameters.

# TODO: add function to check everybody's usage on matrix. same thing for
# lithium and matrix.

# TODO: functions to inspect the mistakes made by the model.

# NOTE: some of these consistency checks can be done automatically.

# TODO: something to randomize the training data, and to keep some set of sentences
# aligned.
# TODO: some tests like checking agreement of dimensions.
# or for paired iteration: that is pretty much just izip or something like that.

# TODO: another test that may be worth validating is that the model,
# may have some property that must be satisfied between pairs of the models.
# using stuff in terms of dictionaries is nice.

# TODO: differential testing, given the same prediction files, are there mistakes
# that one model makes that the other does not.
# having a function to check this is interesting.

# stuff to go back and train on all data. check this.

### some useful assert checks.
# NOTE: this is recursive, and should be applied only to sequences
# of examples.

# TOOD: something that can be computed per length.

### TODO: another important thing is managing the experiments.
# this means that

# TODO: perhaps important to keep a few torch models
# and stuff.

# TODO: the Tensorflow models can be kept in a different file.

# TODO: add code to deal with the exploration of the results.
# TODO: also some tools for manipulation with LSTMs.

# TODO: stuff to build ensembles easily.

# TODO: add stuff for initial step size tuning.

# TODO: add functionality to run DeepArchitect on the matrix server.

# TODO: add function doing cross validation, that is then ran on the
# full dataset.

# TODO: batchification

# TODO: stuff for cross validation . it has to be more general than the
# stuff that was done in scikit-learn.

# are there things that are common to all models and all experiments.

# handling multiple datasets is interesting. what can be done there?

# how to manage main files and options more effectively.

# TODO: it is interesting to think about how can the featurizers be applied to
# a sequence context. that is more tricky. think about CRFs and stuff like that.

# TODO: think about an easy map of neural network code to C code, or even to
# more efficient python code, but that is tricky.

# think about when does something become so slow that it turns impractical?

# TODO: look into how to use multi-gpu code.

# TODO: tools for masking and batching for sequence models and stuff.
# these are interesting.

# check mongo db or something like that, and see if it is worth it.

# how would beam search look on Tensorflow, it would require executing with the
# current parameters, and the graph would have to be fixed.

# possible to add words in some cases with small descriptions to the experiments.

# by default, the description can be the config name.

# the only thing that how approach changes is the way the model is trained.

# beam search only changes model training.

# is it possible to use the logger to save information to generate results or
# graphs.

# in Tensorflow and torch, models essentially communicate through
# variables that they pass around.

# for trading of computation and safety of predictions.
# check that this is in fact possible. multi-pass predictions.

# TODO: an interesting interface for many things with state is the
# step; and query functions that overall seem to capture what we care about.

# TODO: encode, decode, step,
# I think that this is actually a good way of putting this interface.

# TODO: have something very standard to sweep learning rates.
# what can be done here?

# NOTE: it is not desirable for the rate counter to always return a step size,
# because the optimizers have state which can be thrown away in the case of a
# TODO: stuff to deal with unknown tokens.
# TODO: sort of the best performance of the model is when it works with

# there is all this logic about training that needs to be reused.
# this is some nice analysis there.

# some of the stuff that I mention for building networks.

## TODO: interface for encoder decoder models.

# TODO: stuff that handles LSTM masking and unmasking.

# pack and unpack. functionality. keeping track of the batch dimension.

# use of map and reduce with hidden layers.

# TODO: perhaps the results can be put in the end of the model.

# padding seems to be done to some degree by the other model.

# TODO: the notion of a replay settings. a set of variables that is kept track
# during various places.

# TODO: notion of driving training, and keeping some set of variables
# around that allow me to do that.

# keeping the stuff way is a good way of not cluttering the output.

# batching is quite important, and I need to do something that easily goes
# from sequences to batches.

# there is a problem with the resets.
# once it resets, you have to consider where did it reset to, so you can
# reduce the step size from there.
# basically, resetting gets much harder.

# TODO: basically switch between the format of column and row for sequences
# of dictionaries. this is going to be simple. also, can just provide an
# interface for this.

# TODO: checkpoints with resetting seem like a nice pattern to the model.

# TODO: extend this information about the model to make sure that the model
# is going to look the way

# it is going to be nice to extend the config generator to make sure that
# things work out. for example, different number of arguments, or rather
# just sequence multiple config generators or something like that.

# that is simple to do. it is just a matter of extending a lot of stuff.

# improve debugging messages because this is hard to read.

# TODO: add some plotting scripts for typical things like training and
# validation error. make sure that this is easy to do from the checkpoints and
# stuff.

# TODO: perhaps for the most part, these models could be independent.
# this means that they would not be a big deal. this is going to be interesting.

# also, the notion of copying something or serializing it to disk is relevant.

# managing a lot of hyperparameters that keep growing is troublesome.
# might also be a good idea to use a tuner.

# TODO: clean for file creation and deletion, this may be useful to group
# common operations.

# TODO: registering various clean up operations and stuff like that.
# that can be done at teh
# TODO: tests through config validation.

# TODO: possibility of running multiple times and getting the median.

# TODO: what is a good way of looking at different results of the model.
# it would be interesting to consider the case

# TODO: those runs about rate patiences are the right way of doing things.
# TODO: add easy support for running once and then running much slower.

# TODO: rates make a large difference.

# TODO: add stuff that forces the reloading of all modules.

# TODO: equivalence between training settings.

# TODO: also, for exploration of different models. what is important to
# check layer by layer.

# TODO: visualization of different parts of the model, and statistics. this is
# going to be interesting. like exploring how different parts of the model.
# change with training time.

# NOTE: things that you do once per epoch do not really matter, as they will be
# probably very cheap computationally comparing to the epoch.

# TODO: support for differential programming trained end to end.

# TODO: even to find a good configuration, it should be possible to figure
# out reasonable configurations by hand.

# equivalence tests. this running multiple configs that you know that
# should give the same result.

# TODO: it may be worth to come with some form of prefix code for the experiments
# it may be worth to sort them according to the most recent.

# TODO: some stuff to do an ensemble. it should be simple. just average the
# predictions of the top models, or do some form of weighting.

# it should be trivial to do, perhaps stack a few of these images, or tie
# the weights completely. maybe not necessary.

# TODO: processing the data is going to be interesting.

# TODO: stuff for easily doing ensembles of models.

# if just based on full predictions, that is fine, but if transitioning
# requires running the model, it gets a bit more complicated. multi-gpu
# options are possible, and doable. it is a matter of model serving.

# TODO: throw away 5 percent of the data where you make the most mistakes and
# retrain. perhaps these are training mistakes.

# transpose a nested dict.

# TODO: several step functions to featurize, and perhaps visualize the
# results of that model.

# TODO: totally add stuff for cross validation.
# use the same train and test splits.
# TODO: perhaps base it around iterables.

# TODO: do train_dev split in a reasonable way.
# perhaps with indices. how did I do it in CONLL-2000 integrate that.

# TODO: for the seed add support for other things that may have independent
# seeds.

# TODO: can add hyperparameters for the data loading too. it
# it just get more complicated.

# TODO: look at writing reentrant code based on checkpoint.json

# TODO: allow aliasing in some configurations. what is the right way
# of going about this. I can be done in the experiments preprocessing part.

# NOTE: the aliasing idea for configs is quite nice.

# NOTE: better logging and accumulation of information.
# I think that this can be done through registering, and passing
# information that uses that registered function to do something.
# it is quite clean like this.

# TODO: stuff to maintain aliasing, or to run alias in slightly different
# configurations; e.g., sharing some parameters, but having the other ones
# fixed to some relevant specification.

# for managing experiments and comparing.
# to develop something for experiment management, it it is convenient to have
# a way of interacting with them online. this would make it.

# add information for each time a logging event of a given type is called.

# also add the option to do things to the terminal.
# it would be convenient to make sure that  I can debug a model in such a way
# that it makes it easy to look at differential parts of the score.
# what to print in the terminal. there is also the question about suppression,
# in the beginning,. the probes that we can use make sense.

# it would be nice to have a way of deactivating the logging somehow.

# TODO: to get configs that have variable number of arguments, it has to be
# possible to specify what are the options that are defining the arguments that we care about.
# only if these are set, will those arguments be available.

# the fact that the configs are variable will mean that I will have to pass a
# dictionary to the create_experiment function.

# TODO: it should be possible to call a function with a dictionary with
# a superset of the arguments of a function.
# if that function has defaults, use those if those elements are not in the
# dictionary.
# this is mostly for convenience, for functions that have a lot of
# arguments. it may be important to structure and unstructure a dictionary
# to retrieve these types of arguments, because it gets really messy.
# namespace for arguments.

# NOTE: certain things should be more compositional.
# TODO: perhaps add the is sparse information.
# there is information about the unknown tokens and stuff like that?
# there are things that can be done through indexing.
# can featurize a set of objects, perhaps.
# do it directly to each of the xs passed.

# the construction of the dictionary can be done incrementally.

# this should handle integer featurization, string featurization, float
# featurization

# subset featurizers, and simple ways of featurizing models.

# features from side information.
# this is kind of a bipartite graph.
# each field may have different featurizers
# one field may have multiple featurizers
# one featurizers may be used in multiple fields
# one featurizer may be used across fields.
# featurizers may have a set of fields, and should be able to handle these
# easily.

# features from history.
# features from different types of data. it should be easy to integrate.

# easy to featurize sets of elements,

# handling gazeteers and stuff like that.
# it is essentially a dictionary.
# there is stuff that I can do through

# I can also register functions that take something and compute something.
# and the feature is that. work around simple feature types.

# come up with some reasonable interface for featurizers.
# TODO: have a few operations defined on featurizers.

# NOTE: that there may exist different featurizers.
# NOTE: I can have some form of function that guesses the types of elements.

# NOTE: for each CSV field, I can register multiple features.

# NOTE that this is mainly for a row. what about for things with state.

# compressing and decompressing folders.

# TODO: add stuff to do animations easily.

# TODO: add plotting functionality to generate grid plots easily.

# what is the difference between axis and figures. plot, axis, figures

# NOTE: for example, you can keep the labels and do something with the
# the rest of the model.
# you can do a lot of thing.

# another type of graph that is common.
# which is.

# TODO: log plots vs non log plots. more properties to change.
# TODO: updatable figure.

# # TODO: also have some way of adding

# subplots with animation. that would be nice. multiple animations side by side.
# rather than classes, it may be worth

# NOTE: this is going to be done in the head node of the servers for now.
# NOTE: may return information about the different files.
# may do something with the verbose setting.

# TODO: this can be more sophisticated to make sure that I can run this
# by just getting a subset of the files that are in the source directory.
# there is also questions about how does this interact with the other file
# management tools.

# NOTE: this might be useful but for the thing that I have in mind, these
# are too many options.
# NOTE: this is not being used yet.

# there is the override and transfer everything, and remove everything.

# NOTE: the use case is mostly to transfer stuff that exists somewhere
# from stuff that does not exist.

# there is compression stuff and other stuff.

# NOTE: the stuff above is useful, but it  is a little bit too much.
# only transfer newer.

# delete on destination and stuff like that. I think that these are too many
# options.

# remote deletes are probably reasonable.

# imagine that keeps stuff that should not been there.
# can also, remove the folder and start from scratch
# this is kind of tricky to get right. for now, let us just assume that

