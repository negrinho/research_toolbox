collection of utils that are widely used to run machine learning experiments and in training models. I intend to  keep some information about the design of these things.


eval and train should be different from each other.
I guess eval only needs the output of the model
def should know how to run the model.

this would be the predict part.

model_def can be seem as train only actually.

there is also model serialization.

separate phases 

loading 
training
evaluating
analyzing

model serialization should be easy

identify the different factors of variation. 

should be easy to try a different optimization algorithm. 
should be easy to try a different model 
should be easy to try a different dataset 
should be easy to try a different metric

should be reproducible
should be extensible <- 
should be easy to get started with
should be easy to get information out of the model 

very easy parallelization through ids. 
this is simple to do.

extracting results is important to do. 
what is a nice way of doing this.

need to be done in a single file, assuming that the underlying file did not
change. It may be useful ot keep the file that was run somewhere.

procedural generated experiments. 
tool to explore the results of experiments

has to be through the top level. only uses what you feel necessary. 
code should not be affected.

typical slices with respect to a reference point. 
all in one dimension.

all possible changes to two dimensions
all possible changes to all dimensions. 

this can definitely be done. easy to do things in this setting.

have a way of having holes in the hyperparameters.

show types and possibilities for each set.

check nproc and stuff like that.

easy way of getting notified by email when things terminate.

use the server more effectively.

prefix experiments with date and git name.